"""Goal and path chain acceptance (order section 13.4-A).

The question this settles is narrow and was previously answered by guesswork:
when nothing is in the way, does the controller actually enter the common
0.25 m goal region, or does it stabilise short of it?

`Contouring::isObjectiveReached` uses a 1.0 m threshold, but the bridge never
calls it (verified: zero occurrences in the bridge's call chain), so nothing in
this pipeline lets a method declare victory early.  Whatever stopping happens is
the controller's own convergence, not a termination rule.

Three probes, all obstacle-free so that only the goal/path chain is exercised:

  1. long and short approaches from several initial headings;
  2. a start already inside 1.0 m of the goal -- the last metre on its own;
  3. an obstacle that is present and then removed, to see whether the controller
     resumes and closes the distance once the constraint disappears.

For each run the first entry time into 0.25 / 0.5 / 1.0 m is recorded together
with whether the trajectory was collision-free up to that moment, which is the
auxiliary criterion the order requires be reported for every method alike.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

RADII = (0.25, 0.5, 1.0)


def open_bridge(binary, settings):
    return subprocess.Popen([binary, settings], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)


def talk(proc, line):
    proc.stdin.write(line + "\n")
    proc.stdin.flush()
    reply = proc.stdout.readline().strip()
    if not reply:
        raise RuntimeError("bridge closed the connection")
    return reply


def reference(start, goal, points, overshoot):
    direction = (goal[0] - start[0], goal[1] - start[1])
    span = math.hypot(*direction)
    if span > 1e-9 and overshoot:
        goal = (goal[0] + direction[0] / span * overshoot,
                goal[1] + direction[1] / span * overshoot)
    return [(start[0] + (goal[0] - start[0]) * i / (points - 1),
             start[1] + (goal[1] - start[1]) * i / (points - 1))
            for i in range(points)]


def step_line(state, goal, path, obstacles, horizon, dt):
    parts = ["STEP", *(f"{v:.9f}" for v in state), f"{goal[0]:.9f}", f"{goal[1]:.9f}",
             "NPATH", str(len(path))]
    for px, py in path:
        parts += [f"{px:.9f}", f"{py:.9f}"]
    parts += ["NOBS", str(len(obstacles))]
    for i, (ox, oy, radius, sigma) in enumerate(obstacles):
        parts += [str(i), f"{ox:.9f}", f"{oy:.9f}", "0.0", f"{radius:.9f}",
                  "NPRED", str(horizon)]
        for _ in range(horizon):
            parts += [f"{ox:.9f}", f"{oy:.9f}", "0.0", f"{sigma:.9f}", f"{sigma:.9f}"]
    return " ".join(parts)


def drive(proc, horizon, dt, start, heading, goal, steps, obstacle_until=None,
          obstacle=None, overshoot=2.0, robot_radius=0.3):
    """Returns first-entry times and whether the prefix was collision-free."""
    talk(proc, "RESET")
    path = reference(start, goal, 8, overshoot)
    state = [start[0], start[1], heading, 0.0]
    first = {r: None for r in RADII}
    clean = {r: True for r in RADII}
    collided = False
    for step in range(steps):
        active = []
        if obstacle is not None and (obstacle_until is None or step < obstacle_until):
            active = [obstacle]
        reply = talk(proc, step_line(state, goal, path, active, horizon, dt)).split()
        if reply[0] != "STEP":
            raise RuntimeError(f"bridge error: {' '.join(reply[:4])}")
        v, w = float(reply[2]), float(reply[3])
        state[2] += w * dt
        state[3] = v
        state[0] += v * math.cos(state[2]) * dt
        state[1] += v * math.sin(state[2]) * dt
        for ox, oy, radius, _ in active:
            if math.hypot(state[0] - ox, state[1] - oy) - robot_radius - radius < 0.0:
                collided = True
        distance = math.hypot(state[0] - goal[0], state[1] - goal[1])
        for r in RADII:
            if first[r] is None and distance <= r:
                first[r] = (step + 1) * dt
                clean[r] = not collided
        if first[min(RADII)] is not None:
            break
    return {"first_entry": first, "clean_prefix": clean,
            "final_distance": math.hypot(state[0] - goal[0], state[1] - goal[1]),
            "steps": step + 1}


def main():
    workspaces = {"tmpc": "/home/abc/workspace/bayes_occ_mpc/build_modern/tmpc",
                  "shmpc": "/home/abc/workspace/bayes_occ_mpc/build_modern/shmpc"}
    results = {}
    print(f"{'臂':7s} {'探针':22s} {'0.25m':>7s} {'0.5m':>7s} {'1.0m':>7s} "
          f"{'终点距离':>9s} {'步数':>5s}")
    for arm, workspace in workspaces.items():
        binary = f"{workspace}/devel/lib/mpc_planner_bridge/mpc_bridge"
        settings = (f"{workspace}/src/mpc_planner/mpc_planner_jackalsimulator"
                    f"/config/settings.yaml")
        proc = open_bridge(binary, settings)
        info = talk(proc, "INFO").split()
        horizon, dt = int(info[1]), float(info[2])
        arm_results = {}
        probes = [
            ("长距离 heading=+pi/2", (0.0, -4.0), math.pi / 2, (0.0, 4.0), 200, None, None),
            ("长距离 heading=0", (0.0, -4.0), 0.0, (0.0, 4.0), 200, None, None),
            ("长距离 heading=-pi/2", (0.0, -4.0), -math.pi / 2, (0.0, 4.0), 200, None, None),
            ("短距离 1.5m", (0.0, -1.5), math.pi / 2, (0.0, 0.0), 120, None, None),
            ("最后一米 0.9m", (0.0, -0.9), math.pi / 2, (0.0, 0.0), 120, None, None),
            ("障碍移开后继续", (0.0, -4.0), math.pi / 2, (0.0, 4.0), 200, 40,
             (0.0, 0.0, 0.3, 0.4)),
        ]
        for name, start, heading, goal, steps, until, obstacle in probes:
            try:
                out = drive(proc, horizon, dt, start, heading, goal, steps,
                            obstacle_until=until, obstacle=obstacle)
            except Exception as exc:
                out = {"error": f"{type(exc).__name__}: {exc}"}
                proc = open_bridge(binary, settings)
                talk(proc, "INFO")
            arm_results[name] = out
            if "error" in out:
                print(f"{arm:7s} {name:22s} 失败 {out['error'][:40]}")
                continue
            fe = out["first_entry"]
            cell = lambda r: (f"{fe[r]:.2f}s" if fe[r] is not None else "—")
            print(f"{arm:7s} {name:22s} {cell(0.25):>7s} {cell(0.5):>7s} {cell(1.0):>7s} "
                  f"{out['final_distance']:9.2f} {out['steps']:5d}", flush=True)
        talk(proc, "QUIT")
        proc.wait(timeout=30)
        results[arm] = arm_results

    Path("/home/abc/temp/modern/snapshot/goal_audit.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    print("\n判据：`isObjectiveReached` 在 bridge 调用链上出现 0 次，")
    print("      因此没有任何方法能在本回路里提前宣布到达；未进入 0.25 m 即为未收敛。")
    print("明细 -> /home/abc/temp/modern/snapshot/goal_audit.json")


if __name__ == "__main__":
    main()
