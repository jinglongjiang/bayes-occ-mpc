"""Frozen, resumable evaluation of the repaired native MPC comparators.

Development uses five pedestrians only. Selection never reads test outcomes;
every native episode starts a fresh controller with an explicit guidance seed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np

import modern_repair as repair
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env
from evaluate_matched_safety import SCENES

OUT = Path("/home/abc/temp/modern/repaired_main")
DEV = [("dev_circle", 5, "circle_crossing", 4., 10.),
       ("dev_square", 5, "square_crossing", 4., 10.)]
FAMILIES = ("bayes", "tmpc", "shmpc")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def layout(env):
    agents = [env.robot, *env.humans]
    values = np.asarray([[a.px, a.py, a.gx, a.gy, a.v_pref, a.radius] for a in agents], dtype=np.float64)
    return hashlib.sha256(values.round(9).tobytes()).hexdigest()


def candidates():
    result = []
    for horizon in (8, 16):
        for point in range(5):
            result.append(dict(id=f"bayes_h{horizon}_p{point}", family="bayes", arm="bayes",
                               horizon=horizon, point=point, profile="legacy", risk=None))
        for family in ("tmpc", "shmpc"):
            risks = (.05, .35, .65, .90) if family == "tmpc" else (.025, .175, .325, .45)
            for profile in ("legacy", "track", "goal_track"):
                for risk in risks:
                    result.append(dict(id=f"{family}_h{horizon}_{profile}_r{risk:g}", family=family,
                        arm=f"{family}_repair_n{horizon}", horizon=horizon, point=None,
                        profile=profile, risk=risk))
    return result


def gather_history():
    used, hashes, files = set(), set(), 0

    def harvest(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("case_id"), int):
                used.add(obj["case_id"])
            if isinstance(obj.get("cases"), list):
                used.update(i for i in obj["cases"] if isinstance(i, int))
            for key, value in obj.items():
                if key.startswith("layout_hash") or key.startswith("layout_sha"):
                    vals = value.values() if isinstance(value, dict) else [value]
                    hashes.update(v for v in vals if isinstance(v, str))
                harvest(value)
        elif isinstance(obj, list):
            for value in obj:
                harvest(value)

    seen = set()
    for root in (Path("/home/abc/temp"), Path("/home/abc/workspace/bayes_occ_mpc")):
        for directory, children, names in os.walk(root):
            children[:] = [n for n in children if n not in
                           (".git", "__pycache__", "build_modern", "node_modules", ".venv")]
            if Path(directory) == OUT or OUT in Path(directory).parents:
                children[:] = []
                continue
            for name in names:
                path = Path(directory) / name
                if path.suffix not in (".json", ".jsonl") or path.resolve() in seen:
                    continue
                seen.add(path.resolve())
                try:
                    with path.open() as handle:
                        if path.suffix == ".jsonl":
                            for line in handle:
                                if line.strip():
                                    harvest(json.loads(line))
                        else:
                            harvest(json.load(handle))
                    files += 1
                except (ValueError, OSError):
                    continue
    return used, hashes, files


def prepare():
    if (OUT / "protocol.json").exists():
        return json.loads((OUT / "protocol.json").read_text())
    used, historical_hashes, files = gather_history()
    reserved = set(range(20000, 20020)) | set(range(20100, 20200)) | set(range(21000, 21100)) | set(range(21200, 21205))
    if used & reserved:
        raise RuntimeError(f"reserved cases previously used: {sorted(used & reserved)}")
    registry, hashes_by_split = {}, {}
    plans = [("D1", DEV, range(20000, 20020)), ("D2", DEV, range(20100, 20200)),
             ("T", SCENES, range(21000, 21100)), ("L", SCENES, range(21200, 21205))]
    for split, scenes, cases in plans:
        hashes_by_split[split] = set()
        for scene in scenes:
            name, n, generator, radius, width = scene
            env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", n, generator, radius, width, 25, True)
            for case in cases:
                env.reset(options={"test_case": case})
                h = layout(env)
                if h in hashes_by_split[split] or h in historical_hashes or h[:16] in historical_hashes:
                    raise RuntimeError(f"layout reuse: {split}/{name}/{case}")
                hashes_by_split[split].add(h)
                registry[f"{split}/{name}/{case}"] = h
    for a in hashes_by_split:
        for b in hashes_by_split:
            if a != b and hashes_by_split[a] & hashes_by_split[b]:
                raise RuntimeError(f"layout overlap: {a}/{b}")
    protocol = dict(created=time.time(), history_files=files, history_case_count=len(used),
        history_max_case=max(used), candidates=candidates(), layouts=registry,
        task=dict(robot_visible=False, v_max=1., a_max=2., omega_max=.8, dt=.25,
                  goal_radius=.3, deadline=25., dynamics="continuous accelerating unicycle"),
        development="five people; D1 10 circle + 10 square; D2 50 circle + 50 square",
        selection="D1 top 3 per family; D2 highest audited SR, then lowest audited CR, then lowest penalty time, then candidate id",
        test="six scenes, 100 untouched layouts/scene, seed 0; both observation conditions",
        stability="occluded first 20 test layouts/scene, additional planner seeds 1 and 2",
        resources=dict(workers=2, thread_limit_per_worker=4, latency_workers=1),
        primary_comparisons=["bayes vs tmpc SR", "bayes vs shmpc SR"],
        risk_note="parameters refer to different risk events; no numerical equality of budgets is asserted",
        label="shared-posterior controllers with documented acados backend repairs; not an end-to-end belief comparison")
    repair.write_result(OUT / "protocol.json", protocol)
    return protocol


def task(split, candidate, scene, case, seed=0, occluded=True):
    body = dict(split=split, candidate=candidate, scene=list(scene), case=case,
                seed=seed, occluded=occluded)
    body["key"] = digest(body)
    return body


def execute(task_data):
    key = task_data["key"]
    target = OUT / "episodes" / f"{key}.json"
    if target.exists():
        record = json.loads(target.read_text())
        if record["task"] != task_data:
            raise RuntimeError("cached task mismatch")
        return record
    os.environ.update(OMP_NUM_THREADS="4", OMP_THREAD_LIMIT="4", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    repair.ROOT = OUT / "raw"
    log_path = OUT / "logs" / f"{key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    c = task_data["candidate"]
    name, n, generator, radius, width = task_data["scene"]
    began = time.monotonic()
    with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        row = repair.cohort([c["arm"]], 1, task_data["case"], n, generator,
            task_data["occluded"], c["profile"], True, radius, width, name,
            native_risk=c["risk"], bayes_point=c["point"] if c["point"] is not None else 4,
            planner_seed_offset=task_data["seed"], config_overrides={"horizon": c["horizon"]})[0]
    env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", n, generator, radius, width, 25, task_data["occluded"])
    env.reset(options={"test_case": task_data["case"]})
    actual_hash = layout(env)
    split = "T" if task_data["split"] in ("T", "S") else task_data["split"]
    protocol = json.loads((OUT / "protocol.json").read_text())
    if actual_hash != protocol["layouts"][f"{split}/{name}/{task_data['case']}"]:
        raise RuntimeError("runtime layout differs from registered layout")
    row.update(development_only=split != "T", evaluation_split=task_data["split"],
               layout_sha256=actual_hash)
    record = dict(task=task_data, result=row, wall_seconds=time.monotonic()-began)
    repair.write_result(target, record)
    return record


def run_tasks(tasks, workers=2):
    if len({t["key"] for t in tasks}) != len(tasks):
        raise RuntimeError("duplicate tasks")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(execute, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            if len(results) % 20 == 0 or len(results) == len(tasks):
                split = tasks[0]["split"]
                repair.write_result(OUT / "status.json", dict(stage=split, complete=len(results), total=len(tasks), updated=time.time()))
                print(f"{split} {len(results)}/{len(tasks)}", flush=True)
    return results


def rank(records):
    groups = {}
    for record in records:
        groups.setdefault(record["task"]["candidate"]["id"], []).append(record["result"])
    def score(item):
        key, rows = item
        return (-np.mean([r["success_without_overlap"] for r in rows]),
                np.mean([r["collision_union"] for r in rows]),
                np.mean([r["nav_time"] if r["success_without_overlap"] else 25. for r in rows]), key)
    return [key for key, _ in sorted(groups.items(), key=score)]


def source_manifest(selected):
    from evaluate_matched_safety import config
    files = {}
    for c in selected:
        controller = None
        repair.runtime_environment()
        try:
            cfg = config(c["point"] or 0)
            if c["family"] != "bayes":
                path, settings = repair.settings(c["arm"], c["profile"], out=OUT, risk=c["risk"])
                controller = repair.JointBridgeController(c["arm"], c["horizon"], .25, settings=str(path), guidance_seed=0)
                cfg = settings
            files.update(repair.provenance(controller, cfg)["files"])
        finally:
            if controller is not None:
                controller.close()
    files[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    addendum = OUT / "PROTOCOL_ADDENDUM.md"
    files[str(addendum)] = hashlib.sha256(addendum.read_bytes()).hexdigest()
    return files


def verify_sources(files):
    changed = [p for p, sha in files.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha]
    if changed:
        raise RuntimeError(f"frozen code/config/binary changed: {changed[:10]}")


def run(stage):
    protocol = prepare()
    catalogue = {c["id"]: c for c in protocol["candidates"]}
    if stage in ("develop", "all"):
        d1 = run_tasks([task("D1", c, DEV[i // 10], 20000+i) for c in catalogue.values() for i in range(20)])
        shortlist = []
        for family in FAMILIES:
            shortlist += [catalogue[k] for k in rank([r for r in d1 if r["task"]["candidate"]["family"] == family])[:3]]
        repair.write_result(OUT / "shortlist.json", shortlist)
        d2 = run_tasks([task("D2", c, DEV[i // 50], 20100+i) for c in shortlist for i in range(100)])
        selected = [catalogue[rank([r for r in d2 if r["task"]["candidate"]["family"] == family])[0]] for family in FAMILIES]
        freeze = dict(protocol_sha256=digest(protocol), selected=selected, sources=source_manifest(selected), created=time.time())
        repair.write_result(OUT / "frozen.json", freeze)
        print("FROZEN", selected, flush=True)
    if stage in ("test", "all"):
        frozen = json.loads((OUT / "frozen.json").read_text())
        verify_sources(frozen["sources"])
        selected = frozen["selected"]
        results = run_tasks([task("T", c, s, case, occluded=occ) for occ in (True, False)
            for s in SCENES for case in range(21000, 21100) for c in selected])
        verify_sources(frozen["sources"])
        repair.write_result(OUT / "test_results.json", results)
        stability = run_tasks([task("S", c, s, case, seed=seed) for seed in (1, 2)
            for s in SCENES for case in range(21000, 21020) for c in selected])
        repair.write_result(OUT / "stability_results.json", stability)
        latency = run_tasks([task("L", c, s, case) for s in SCENES for case in range(21200, 21205)
                             for c in selected], workers=1)
        repair.write_result(OUT / "latency_results.json", latency)
        verify_sources(frozen["sources"])
        repair.write_result(OUT / "status.json", dict(stage="TEST_COMPLETE", updated=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "develop", "test", "all"])
    args = parser.parse_args()
    run(args.stage)
