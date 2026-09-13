"""The single formal entry point: a task-list driven runner (order section 13.7).

Everything the evaluation runs -- development sweeps and the formal matrix
alike -- goes through this file.  It replaces the earlier ad-hoc launchers and,
in particular, the practice of monkey-patching `BridgeController.act` from a
shell heredoc, which made it impossible to say afterwards what code produced a
number.

Four subcommands, in the order they are used:

    plan       write the task list for a queue (D1, D2, T-OCC, T-FULL, L)
    precheck   verify every task can start: binaries, settings, layouts, splits
    run        execute outstanding blocks, resumable, N workers
    aggregate  recompute the summary from the per-episode records

A block is 25 cases of one (arm, scene, condition, repeat).  Blocks are the unit
of atomic save and of resumption: a block's records are written to a temporary
file and renamed only once the block passes its own completeness check, so an
interrupted run never leaves a half-block that later looks finished.

The bridge process is kept alive across the episodes of a block.  The order
requires the reset cost to be inside the measured loop but forbids paying a
fresh process start-up per formal episode.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import platform
import socket
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

SRC = os.environ.get("SRC", "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, SRC)
sys.path.insert(0, "/home/abc/temp/modern")

import numpy as np                                                    # noqa: E402
from continuous_mpc_gate import DEFAULT_CROWDNAV, run_episode          # noqa: E402
from evaluate_matched_safety import SCENES, config as point_config     # noqa: E402
from unicycle_mpc_gate import UnicycleCEMMPC, unicycle_config          # noqa: E402
from modern_worker import bridge_planner_factory                       # noqa: E402
from candidates import CANDIDATES, settings_for                        # noqa: E402
from analysis import episode_values                                  # noqa: E402

ROOT = Path("/home/abc/temp/modern")
REGISTRY = json.loads((ROOT / "snapshot/case_registry.json").read_text())
BLOCK = 25

# Scenes, keyed the way the registry keys them.
SCENE_BY_ID = {s[0]: tuple(s) for s in SCENES}
SCENE_BY_ID["dev_circle"] = ("dev_circle", 5, "circle_crossing", 4.0, None)
SCENE_BY_ID["dev_square"] = ("dev_square", 5, "square_crossing", None, 10.0)


# --------------------------------------------------------------------- tasks

def _blocks(cases):
    return [cases[i:i + BLOCK] for i in range(0, len(cases), BLOCK)]


def _split_cases(name):
    return REGISTRY["splits"][name]["cases"]


def plan_queue(queue: str):
    """Return the task list for one queue.  Deterministic: the same call always
    produces the same blocks in the same order, so a resumed run and a fresh one
    address identical work."""
    tasks = []

    def add(arm, family, candidate_id, scene_id, cases, condition, repeat,
            suffix="", **extra):
        for index, chunk in enumerate(_blocks(cases)):
            tasks.append({
                "block_id": f"{queue}|{arm}|{scene_id}|{condition}|r{repeat}|b{index}{suffix}",
                "queue": queue, "arm": arm, "family": family,
                "candidate_id": candidate_id, "scene_id": scene_id,
                "condition": condition, "repeat": repeat, "cases": chunk,
                **extra,
            })

    if queue == "D1":
        # 12 candidates x 40 five-person layouts, one planning repeat.
        for family, rows in CANDIDATES.items():
            for candidate in rows:
                for split, scene_id in (("D1_circle", "dev_circle"),
                                        ("D1_square", "dev_square")):
                    add(f"{family}:{candidate.candidate_id}", family,
                        candidate.candidate_id, scene_id, _split_cases(split),
                        "occluded", 0)
    elif queue == "D2":
        selected = json.loads((ROOT / "snapshot/d1_selection.json").read_text())
        # A family missing from the selection means D1 did not finish for it.
        # Planning fewer arms would silently shrink the comparison, so this is
        # an error rather than something to work around.
        missing = [f for f in CANDIDATES if f not in selected["selected"]]
        if missing:
            raise SystemExit(
                f"D1 选点缺少方法家族 {missing}；先跑完 D1 再规划 D2")
        for family, chosen in selected["selected"].items():
            for candidate_id in chosen:
                for repeat in (0, 1):
                    for split, scene_id in (("D2_circle", "dev_circle"),
                                            ("D2_square", "dev_square")):
                        add(f"{family}:{candidate_id}", family, candidate_id,
                            scene_id, _split_cases(split), "occluded", repeat)
    elif queue in ("T-OCC", "T-FULL"):
        frozen = json.loads((ROOT / "snapshot/frozen.json").read_text())
        condition = "occluded" if queue == "T-OCC" else "full"
        arms = frozen["arms_occluded"] if queue == "T-OCC" else frozen["arms_full"]
        for arm, spec in arms.items():
            for scene_id in SCENE_BY_ID:
                if not scene_id.startswith("dev"):
                    for repeat in (0, 1, 2):
                        add(arm, spec["family"], spec["candidate_id"], scene_id,
                            _split_cases(f"T_{scene_id}"), condition, repeat,
                            existence_override=spec.get("existence_override"))
    elif queue == "L":
        # The exclusive timing window: every arm, every scene, five registered
        # debug layouts, run twice -- once under the same resource budget the
        # formal matrix used, once single-threaded.  Timing is the measurement,
        # so blocks are never run concurrently; the runner is called with one
        # worker.
        frozen = json.loads((ROOT / "snapshot/frozen.json").read_text())
        for arm, spec in frozen["arms_occluded"].items():
            for scene_id in SCENE_BY_ID:
                if scene_id.startswith("dev"):
                    continue
                for threads in (4, 1):
                    add(arm, spec["family"], spec["candidate_id"], scene_id,
                        _split_cases(f"L_{scene_id}"), "occluded", 0,
                        suffix=f"|t{threads}",
                        existence_override=spec.get("existence_override"),
                        thread_config=threads)
    else:
        raise SystemExit(f"unknown queue {queue!r}")
    return tasks


# ------------------------------------------------------------------ planners

class _Persistent:
    """A `planner_type` that hands `run_episode` the same planner every episode.

    `run_episode` constructs its planner by calling `planner_type(config)`.
    Returning an already-running bridge from that call, after resetting it, is
    what keeps one process alive across a block; the reset cost stays inside the
    episode, which is what the order asks be measured.
    """

    def __init__(self, build):
        self._build = build
        self._planner = None

    def __call__(self, config):
        if self._planner is None:
            self._planner = self._build(config)
        else:
            self._planner.reset()
        return self._planner

    def close(self):
        controller = getattr(self._planner, "_controller", None)
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass
        self._planner = None


def make_planner(family: str, candidate_id: str, arm: str,
                 existence_override=None):
    """Returns (persistent planner factory, MPCConfig) for one arm."""
    candidate = next(c for c in CANDIDATES[family] if c.candidate_id == candidate_id)
    if family == "bayes":
        # `bayes_r1` is the same frozen configuration with the planning-side
        # existence probability pinned to 1.0.  `existence_override` is the
        # registered switch for exactly this: it is read where the risk term
        # weights each track and nowhere else, so the tracker's survival, track
        # deletion, the means and the covariances are untouched.  The control
        # arm is never selected separately -- it inherits whatever the Bayesian
        # family froze.
        cfg = unicycle_config(point_config(candidate.risk_point),
                              horizon=candidate.horizon,
                              population=candidate.population,
                              iterations=candidate.iterations,
                              existence_override=existence_override)
        return _Persistent(UnicycleCEMMPC), cfg
    cfg = unicycle_config(point_config(2), horizon=candidate.horizon)
    settings = str(settings_for(candidate))
    build = lambda c, v=candidate.workspace, s=settings, o=candidate.path_overshoot: (
        bridge_planner_factory(v, settings=s, path_overshoot=o)(c))
    return _Persistent(build), cfg


# ------------------------------------------------------------------ execution

def block_path(task):
    safe = task["block_id"].replace("|", "__").replace(":", "-")
    return ROOT / "runs" / task["queue"] / f"{safe}.jsonl"


def block_done(task):
    """A block counts as done only if its file holds one record per case with
    the expected keys -- seeing the file is not enough."""
    path = block_path(task)
    if not path.exists():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except Exception:
        return False
    if len(rows) != len(task["cases"]):
        return False
    if {r["case_id"] for r in rows} != set(task["cases"]):
        return False
    return all(record_valid(r, task) for r in rows)


def record_valid(row, task):
    expected = _layout_hashes_for(task, task["scene_id"])
    fields = ("block_id", "queue", "arm", "family", "candidate_id", "scene_id",
              "condition", "repeat")
    if any(row.get(key) != task[key] for key in fields):
        return False
    if row.get("layout_sha256_16") != expected.get(str(row.get("case_id"))):
        return False
    if task["queue"] in ("T-OCC", "T-FULL", "L"):
        frozen_path = ROOT / "snapshot/frozen.json"
        if not frozen_path.exists() or row.get("frozen_sha256") != hashlib.sha256(
                frozen_path.read_bytes()).hexdigest():
            return False
    if row.get("event") == "error":
        return bool(row.get("failed_attempts"))
    if row.get("event") not in ("reach_goal", "collision", "timeout"):
        return False
    if not all(key in row for key in ("nav_time", "success_without_overlap",
                                      "collision_union", "steps", "code_sha256")):
        return False
    steps = len(row["steps"])
    if not steps or any(len(row.get(key, [])) != steps for key in (
            "plan_step_ms", "pipeline_step_ms")):
        return False
    return True


def validated_rows(queue):
    tasks = plan_queue(queue)
    missing = [task["block_id"] for task in tasks if not block_done(task)]
    if missing:
        raise RuntimeError(f"{queue}: incomplete/invalid blocks ({len(missing)}): {missing[:4]}")
    rows = []
    for task in tasks:
        rows.extend(json.loads(line) for line in block_path(task).read_text().splitlines()
                    if line.strip())
    return rows


def run_block(task, snapshot):
    path = block_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.partial")
    # Preserve interrupted attempts without appending a fresh run to an old prefix.
    for old in (temporary, path):
        if old.exists():
            old.rename(old.with_name(old.name + f".superseded.{time.time_ns()}"))
    scene_id, humans, generator, radius, width = SCENE_BY_ID[task["scene_id"]]
    scene_config_hash = REGISTRY["splits"][
        f"T_{scene_id}" if not scene_id.startswith("dev")
        else ("D1_circle" if scene_id == "dev_circle" else "D1_square")
    ]["scene_config_sha256_16"]
    # The pairing key has to separate the two thread configurations of the
    # timing queue; without it they would collide on the same layout.
    if task.get("thread_config") is not None:
        scene_config_hash = f"{scene_config_hash}-t{task['thread_config']}"
    layout_hashes = _layout_hashes_for(task, scene_id)

    factory, cfg = make_planner(task["family"], task["candidate_id"], task["arm"],
                                task.get("existence_override"))
    rows = []
    started = time.time()
    try:
        for case in task["cases"]:
            row = _run_one(task, factory, cfg, scene_id, humans, generator,
                           radius, width, case, scene_config_hash,
                           layout_hashes.get(str(case)), snapshot)
            rows.append(row)
            with open(temporary, "a") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        factory.close()
    temporary.replace(path)
    if not block_done(task):
        raise RuntimeError(f"invalid completed block: {task['block_id']}")
    return {"block_id": task["block_id"], "n": len(rows),
            "seconds": time.time() - started,
            "errors": sum(1 for r in rows if r["event"] == "error")}


def _layout_hashes_for(task, scene_id):
    if task["queue"] == "L":
        return REGISTRY["splits"][f"L_{scene_id}"]["layout_hashes"]
    if scene_id.startswith("dev"):
        side = "circle" if scene_id == "dev_circle" else "square"
        for split in (f"D1_{side}", f"D2_{side}", f"debug_{side}"):
            entry = REGISTRY["splits"][split]
            if task["cases"][0] in entry["cases"]:
                return entry["layout_hashes"]
        return {}
    return REGISTRY["splits"][f"T_{scene_id}"]["layout_hashes"]


def _run_one(task, factory, cfg, scene_id, humans, generator, radius, width,
             case, scene_config_hash, layout_hash, snapshot):
    """One episode, with one retry reserved for infrastructure failure.

    A repeated algorithmic crash is not retried away: the second failure is
    recorded as an error event with the penalised time, and the collision state
    is reported as unknown rather than assumed clean.
    """
    attempts = []
    for attempt in range(2):
        started = time.perf_counter()
        try:
            signal.alarm(900)
            def verify_layout(env, _observation, _adapter, step):
                if step == 0:
                    from case_registry import layout_hash as observed_hash
                    if observed_hash(env) != layout_hash:
                        raise RuntimeError("runtime initial layout differs from registry")
            result = run_episode(
                DEFAULT_CROWDNAV, "bayes", humans, generator, case, cfg,
                radius, width, task["repeat"], 0.0, 0.0, 1.0, time_limit=25,
                planner_type=factory, robot_kinematics="unicycle",
                step_observer=verify_layout,
                occlusion=(task["condition"] == "occluded"))
            row = asdict(result)
            row.update(_meta(task, scene_id, case, scene_config_hash,
                             layout_hash, cfg, snapshot))
            row["wall_s"] = time.perf_counter() - started
            row["attempts"] = attempt + 1
            row["failed_attempts"] = attempts
            return row
        except Exception as exc:
            attempts.append({"type": type(exc).__name__, "message": str(exc)[:400],
                             "traceback": traceback.format_exc()[-1500:]})
            factory.close()
        finally:
            signal.alarm(0)
    row = {"event": "error", "success": 0, "collision": 0, "timeout": 0,
           "success_without_overlap": 0, "collision_union": None,
           "nav_time": 25.0, "scored_time": 25.0, "path_length": 0.0,
           "steps": [], "goal_entry": {}, "goal_entry_clean": {},
           "plan_step_ms": [], "pipeline_step_ms": [],
           "bound_violations": 0, "max_bound_excess": 0.0,
           "attempts": 2, "failed_attempts": attempts}
    row.update(_meta(task, scene_id, case, scene_config_hash, layout_hash, cfg,
                     snapshot))
    return row


def _meta(task, scene_id, case, scene_config_hash, layout_hash, cfg, snapshot):
    return {
        "block_id": task["block_id"], "queue": task["queue"], "arm": task["arm"],
        "family": task["family"], "candidate_id": task["candidate_id"],
        "scene_id": scene_id, "scene_config_sha256_16": scene_config_hash,
        "case_id": case, "env_seed": case, "repeat": task["repeat"],
        "planner_seed_offset": task["repeat"], "condition": task["condition"],
        "layout_sha256_16": layout_hash,
        "pairing_key": "|".join([scene_id, scene_config_hash, str(case),
                                 task["condition"], str(task["repeat"])]),
        "horizon": cfg.horizon, "dt": cfg.dt, "v_max": cfg.v_max,
        "a_max": cfg.a_max, "omega_max": getattr(cfg, "omega_max", None),
        "human_margin": cfg.human_margin,
        "existence_override": getattr(cfg, "existence_override", None),
        "host": snapshot["host"], "threads": snapshot["threads"],
        "code_sha256": snapshot["code_sha256"],
        "frozen_sha256": snapshot.get("frozen_sha256"),
        "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
    }


# ------------------------------------------------------------------- worker

def _worker(argument):
    task, snapshot = argument
    threads = task.get("thread_config", snapshot["threads"])
    snapshot = {**snapshot, "threads": threads}
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[variable] = str(threads)
    def expired(_signum, _frame):
        raise TimeoutError("episode exceeded 900 seconds wall-clock")
    signal.signal(signal.SIGALRM, expired)
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(limits=threads):
            return run_block(task, snapshot)
    except Exception as exc:
        return {"block_id": task["block_id"], "n": 0, "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}


def _snapshot(threads):
    sources = [Path(SRC) / "continuous_mpc_gate.py", Path(SRC) / "unicycle_mpc_gate.py",
               Path(SRC) / "modern_worker.py", ROOT / "candidates.py",
               ROOT / "bridge/run_bridge_cohort.py"]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.read_bytes())
    frozen = ROOT / "snapshot/frozen.json"
    return {"host": socket.gethostname(), "threads": threads,
            "code_sha256": digest.hexdigest()[:16],
            "frozen_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest() if frozen.exists() else None,
            "platform": platform.platform()}


# --------------------------------------------------------------------- main

def cmd_plan(args):
    tasks = plan_queue(args.queue)
    target = ROOT / "tasks" / f"{args.queue}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    episodes = sum(len(t["cases"]) for t in tasks)
    print(f"{args.queue}: {len(tasks)} 个块，{episodes} 回合 -> {target}")


def cmd_precheck(args):
    tasks = [json.loads(l) for l in (ROOT / "tasks" / f"{args.queue}.jsonl").read_text().splitlines() if l.strip()]
    problems = []
    if tasks != plan_queue(args.queue):
        problems.append("task manifest differs from registered plan")
    if args.queue in ("T-OCC", "T-FULL", "L"):
        from freeze import verify_frozen
        verify_frozen()
    seen_arms = set()
    for task in tasks:
        if task["arm"] in seen_arms:
            continue
        seen_arms.add(task["arm"])
        candidate = next((c for c in CANDIDATES[task["family"]]
                          if c.candidate_id == task["candidate_id"]), None)
        if candidate is None:
            problems.append(f"{task['arm']}: 候选 {task['candidate_id']} 不在冻结清单里")
            continue
        if task["family"] != "bayes":
            binary = (Path("/home/abc/workspace/bayes_occ_mpc/build_modern")
                      / candidate.workspace / "devel/lib/mpc_planner_bridge/mpc_bridge")
            if not binary.exists():
                problems.append(f"{task['arm']}: 缺少 {binary}")
            settings = settings_for(candidate)
            if not settings.exists():
                problems.append(f"{task['arm']}: 缺少 {settings}")
    for task in tasks:
        if task["scene_id"] not in SCENE_BY_ID:
            problems.append(f"{task['block_id']}: 未登记场景 {task['scene_id']}")
        if not _layout_hashes_for(task, task["scene_id"]):
            problems.append(f"{task['block_id']}: case 不在任何登记划分里")
    done = sum(1 for t in tasks if block_done(t))
    print(f"{args.queue}: {len(tasks)} 个块，已完成 {done}，待跑 {len(tasks)-done}")
    print(f"涉及 {len(seen_arms)} 个臂；回合合计 {sum(len(t['cases']) for t in tasks)}")
    if problems:
        print("\n问题：")
        for problem in problems[:30]:
            print("  " + problem)
        raise SystemExit(1)
    print("预检通过")


def cmd_run(args):
    cmd_precheck(args)
    tasks = [json.loads(l) for l in (ROOT / "tasks" / f"{args.queue}.jsonl").read_text().splitlines() if l.strip()]
    outstanding = [t for t in tasks if not block_done(t)]
    if args.limit:
        outstanding = outstanding[:args.limit]
    snapshot = _snapshot(args.threads)
    print(f"{args.queue}: {len(tasks)} 块，待跑 {len(outstanding)}，"
          f"{args.workers} 个 worker × {args.threads} 线程")
    if not outstanding:
        return
    started = time.time()
    finished = 0
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "OMP_THREAD_LIMIT"):
        os.environ[variable] = str(args.threads)
    failed = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for report in pool.imap_unordered(_worker, [(t, snapshot) for t in outstanding]):
            finished += 1
            elapsed = time.time() - started
            rate = elapsed / finished
            remaining = (len(outstanding) - finished) * rate
            flag = "失败" if report.get("failed") else f"{report['n']:2d}回合"
            print(f"[{finished:4d}/{len(outstanding)}] {flag} "
                  f"{report['block_id']}  已用{elapsed/60:.1f}min "
                  f"预计剩余{remaining/60:.0f}min", flush=True)
            if report.get("failed"):
                failed.append(report)
                print("    " + report.get("error", ""), flush=True)
    print(f"{args.queue} 本轮结束，用时 {(time.time()-started)/60:.1f} min")
    if failed or any(not block_done(t) for t in outstanding):
        raise SystemExit(f"{args.queue}: {len(failed)} failed blocks; NOT COMPLETE")
    if args.queue in ("T-OCC", "T-FULL", "L"):
        from freeze import verify_frozen
        verify_frozen()


def cmd_aggregate(args):
    rows = validated_rows(args.queue)
    if not rows:
        raise SystemExit(f"{args.queue}: 没有记录")
    target = ROOT / "snapshot" / f"{args.queue}_episodes.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_arm = {}
    for row in rows:
        by_arm.setdefault((row["arm"], row["scene_id"]), []).append(row)
    print(f"{args.queue}: {len(rows)} 回合，{len({r['arm'] for r in rows})} 个臂")
    print(f"{'arm':26s} {'scene':14s} {'n':>4s} {'SR%':>6s} {'CR%':>6s} "
          f"{'TR%':>6s} {'ERR':>4s} {'罚时s':>7s} {'越界':>5s}")
    summary = {}
    for (arm, scene), group in sorted(by_arm.items()):
        n = len(group)
        errors = sum(1 for r in group if r["event"] == "error")
        values = [episode_values(r) for r in group]
        sr, cr, tr = (100 * float(np.mean([v[m] for v in values])) for m in ("sr", "cr", "tr"))
        penalised = float(np.mean([v["time"] for v in values]))
        violations = sum(r.get("bound_violations", 0) for r in group)
        summary[f"{arm}|{scene}"] = {"n": n, "errors": errors, "sr": sr, "cr": cr,
                                     "tr": tr, "penalised_time": penalised,
                                     "cr_unknown": 100 * float(np.mean([v["cr_unknown"] for v in values])),
                                     "bound_violations": violations}
        print(f"{arm:26s} {scene:14s} {n:4d} {sr:6.1f} {cr:6.1f} {tr:6.1f} "
              f"{errors:4d} {penalised:7.2f} {violations:5d}")
    (ROOT / "snapshot" / f"{args.queue}_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n逐回合 -> {target}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("plan", cmd_plan), ("precheck", cmd_precheck),
                          ("run", cmd_run), ("aggregate", cmd_aggregate)):
        p = sub.add_parser(name)
        p.add_argument("queue")
        p.set_defaults(handler=handler)
        if name == "run":
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.command == "run":
        ROOT.joinpath("runs").mkdir(exist_ok=True)
        with open(ROOT / "runs" / f"{args.queue}.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"{args.queue}: another runner holds the lock")
            args.handler(args)
    else:
        args.handler(args)


if __name__ == "__main__":
    main()
