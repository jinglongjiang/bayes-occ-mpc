"""Closed-loop native smoke for the mpc_planner bridge.

A minimal crossing episode driven entirely by the bridge's own controls: the
robot integrates the second-order unicycle the planner was generated for, and
the pedestrians move at constant velocity.  The point is to prove the planner
runs a real loop and returns usable controls, not to score it -- the scenarios,
radii and scoring of the project are not used here.

Usage:
    python3 drive_bridge.py <bridge binary> <settings.yaml> [--humans N] [--steps K]
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time


def reference_line(start, goal, n=20):
    """A straight reference path from start to goal, which is all the planner
    needs in an empty crossing scene."""
    return [(start[0] + (goal[0] - start[0]) * i / (n - 1),
             start[1] + (goal[1] - start[1]) * i / (n - 1)) for i in range(n)]


def circle_humans(count, radius):
    """Pedestrians on a circle walking to the diametrically opposite point,
    the same geometry the project's circle scenarios use."""
    humans = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count + math.pi / 7.0   # offset off the robot's axis
        px, py = radius * math.cos(angle), radius * math.sin(angle)
        speed = 1.0
        vx, vy = -px / radius * speed, -py / radius * speed
        humans.append([px, py, vx, vy])
    return humans


def step_line(state, goal, path, humans, horizon, dt, sigma_rate, radius):
    parts = ["STEP", *(f"{v:.9f}" for v in state), f"{goal[0]:.9f}", f"{goal[1]:.9f}",
             "NPATH", str(len(path))]
    for px, py in path:
        parts += [f"{px:.9f}", f"{py:.9f}"]
    parts += ["NOBS", str(len(humans))]
    for i, (hx, hy, vx, vy) in enumerate(humans):
        angle = math.atan2(vy, vx)
        parts += [str(i), f"{hx:.9f}", f"{hy:.9f}", f"{angle:.9f}", f"{radius:.9f}",
                  "NPRED", str(horizon)]
        for k in range(1, horizon + 1):
            # Constant velocity mean, uncertainty growing linearly with lookahead.
            sigma = sigma_rate * k * dt
            parts += [f"{hx + vx * k * dt:.9f}", f"{hy + vy * k * dt:.9f}",
                      f"{angle:.9f}", f"{sigma:.9f}", f"{sigma:.9f}"]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("settings")
    ap.add_argument("--humans", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--circle-radius", type=float, default=5.0)
    ap.add_argument("--human-radius", type=float, default=0.3)
    ap.add_argument("--robot-radius", type=float, default=0.325)
    ap.add_argument("--sigma-rate", type=float, default=0.1)
    ap.add_argument("--goal-tolerance", type=float, default=0.5)
    args = ap.parse_args()

    proc = subprocess.Popen([args.binary, args.settings],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def ask(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        reply = proc.stdout.readline().strip()
        if not reply:
            raise RuntimeError("bridge closed the connection")
        return reply

    info = ask("INFO").split()
    if info[0] != "INFO":
        raise RuntimeError(f"unexpected INFO reply: {info}")
    horizon, dt, max_obstacles = int(info[1]), float(info[2]), int(info[3])
    print(f"bridge: N={horizon} dt={dt} max_obstacles={max_obstacles}")
    if args.humans > max_obstacles:
        print(f"!! {args.humans} humans exceeds max_obstacles={max_obstacles}; "
              f"the planner will silently drop the surplus", file=sys.stderr)

    print(ask("RESET"))

    start, goal = (-args.circle_radius, 0.0), (args.circle_radius, 0.0)
    path = reference_line(start, goal)
    humans = circle_humans(args.humans, args.circle_radius)
    # The robot occupies one of the circle slots; keep the humans off its start.
    state = [start[0], start[1], 0.0, 0.0]       # x, y, psi, v

    solve_ms, failures, min_clearance = [], 0, float("inf")
    reached, collided = False, False
    t_start = time.time()

    for step in range(args.steps):
        reply = ask(step_line(state, goal, path, humans, horizon, dt,
                              args.sigma_rate, args.human_radius)).split()
        if reply[0] != "STEP":
            raise RuntimeError(f"bridge error: {' '.join(reply)}")
        success, v_cmd, w_cmd, ms = int(reply[1]), float(reply[2]), float(reply[3]), float(reply[4])
        solve_ms.append(ms)
        failures += (success == 0)

        # Second-order unicycle, the model the solver was generated for.
        state[2] += w_cmd * dt
        state[3] = v_cmd
        state[0] += v_cmd * math.cos(state[2]) * dt
        state[1] += v_cmd * math.sin(state[2]) * dt

        for h in humans:
            h[0] += h[2] * dt
            h[1] += h[3] * dt

        clearance = min(math.hypot(state[0] - h[0], state[1] - h[1])
                        - args.robot_radius - args.human_radius for h in humans)
        min_clearance = min(min_clearance, clearance)
        collided |= clearance < 0.0

        if math.hypot(state[0] - goal[0], state[1] - goal[1]) < args.goal_tolerance:
            reached = True
            break

    ask("QUIT")
    proc.wait(timeout=30)

    steps_run = len(solve_ms)
    ordered = sorted(solve_ms)
    def pct(p):
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else float("nan")

    print(f"humans          {args.humans}")
    print(f"steps           {steps_run}")
    print(f"reached goal    {reached}")
    print(f"collision       {collided}   min clearance {min_clearance:.4f} m")
    print(f"solve failures  {failures} / {steps_run}")
    print(f"solve ms        mean {sum(solve_ms)/max(1,steps_run):.2f}  "
          f"p50 {pct(0.50):.2f}  p95 {pct(0.95):.2f}  max {max(solve_ms, default=float('nan')):.2f}")
    print(f"wall seconds    {time.time() - t_start:.2f}")
    return 0 if steps_run else 1


if __name__ == "__main__":
    sys.exit(main())
