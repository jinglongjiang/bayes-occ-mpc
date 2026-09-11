"""Sequential, development-only five-arm prototype with immutable result provenance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import numpy as np

from continuous_mpc_gate import DEFAULT_CROWDNAV, ObservationAdapter, run_episode
from evaluate_matched_safety import config
from modern_dynamics import ContinuousExecutor, ContinuousUnicycleMPC
from spatial_belief import SpatialAdapter
from spatial_risk_mpc import SpatialRiskMPC
from unicycle_mpc_gate import unicycle_config


ROOT = Path(__file__).resolve().parent
ARMS = ("A_original", "B_interface", "C_projected", "D_shape", "E_hard")
SCENES = (("dev_circle5", 5, "circle_crossing", 4., None, 20100),
          ("dev_square20", 20, "square_crossing", None, 10., 21000))
CONDITIONS = {"clean": (0., 0., 1.), "severe": (.1, .2, .8)}
SOURCE_FILES = ("continuous_mpc_gate.py", "bayesian_rfs.py", "modern_dynamics.py",
                "unicycle_mpc_gate.py", "evaluate_matched_safety.py", "spatial_belief.py",
                "spatial_risk_mpc.py", "run_spatial_prototype.py", "test_spatial_prototype.py")


def source_manifest():
    paths = [ROOT / name for name in SOURCE_FILES]
    paths += [DEFAULT_CROWDNAV / name for name in (
        "crowd_sim/envs/crowd_sim.py", "crowd_sim/envs/occlusion_belief.py",
        "crowd_sim/envs/utils/agent.py", "crowd_sim/envs/policy/orca.py",
        "crowd_nav/configs/env.config")]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_one(arm, scene, case, noise, cfg, count, hard_streak, subdivisions, backend):
    adapter_box, planner_box = [], []

    def adapter_factory(*args, **kwargs):
        if arm == "A_original":
            adapter = ObservationAdapter(*args, **kwargs)
        else:
            representation = {"B_interface": "original_gaussian", "C_projected": "projected",
                              "D_shape": "shape", "E_hard": "shape"}[arm]
            adapter = SpatialAdapter(*args, representation=representation, count=count,
                                     hard_streak=hard_streak if arm == "E_hard" else None, **kwargs)
        adapter_box.append(adapter)
        return adapter

    def planner_factory(configuration):
        planner = (ContinuousUnicycleMPC(configuration) if arm == "A_original"
                   else SpatialRiskMPC(configuration, subdivisions=subdivisions,
                                       backend=backend))
        planner_box.append(planner)
        return planner

    snapshots = []

    def observe(env, obs, adapter, step):
        if (len(snapshots) >= 2 or not adapter.reported_ids or
                not np.any(~obs.human_visible) or not hasattr(obs, "posterior_states")):
            return
        snapshots.append(dict(step=step, robot_xy=obs.robot_xy.tolist(),
            robot_velocity=obs.robot_velocity.tolist(), robot_heading=obs.robot_heading,
            track_ids=adapter.reported_ids, visible=obs.human_visible.tolist(),
            existence=obs.human_existence.tolist(), weights=obs.posterior_weights.tolist(),
            posterior_states=obs.posterior_states.tolist(),
            posterior_accelerations=obs.posterior_accelerations.tolist(),
            selected_controls=planner_box[-1].last_controls.tolist()))

    _, people, generator, radius, width, _ = scene
    start = time.monotonic()
    result = run_episode(DEFAULT_CROWDNAV, "bayes", people, generator, case, cfg,
        radius, width, 0, *noise, time_limit=25, planner_type=planner_factory,
        adapter_factory=adapter_factory, robot_kinematics="unicycle", occlusion=True,
        step_executor=ContinuousExecutor(cfg), step_observer=observe)
    body = asdict(result)
    body.update(arm=arm, scene_id=scene[0], case_id=case, wall_seconds=time.monotonic() - start,
        noise=list(noise), diagnostics=dict(getattr(adapter_box[0].rfs, "diagnostics", {})),
        posterior_snapshots=snapshots, development_only=True,
        execution_model="continuous_accelerating_unicycle",
        hard_streak=hard_streak if arm == "E_hard" else None,
        sweep_subdivisions=None if arm == "A_original" else subdivisions,
        sweep_backend=None if arm == "A_original" else backend)
    body["audited_time"] = result.nav_time if result.success_without_overlap and not result.collision_union else 25.
    return finite_json(body)


def summarize(out, protocol):
    rows = [json.loads(path.read_text()) for path in (out / "episodes").glob("*.json")]
    expected = {(arm, scene[0], case, condition)
                for arm in protocol["arms"] for scene in protocol["scenes"]
                for case in range(scene[-1], scene[-1] + protocol["cases_per_scene"])
                for condition in protocol["conditions"]}
    keys = [(r["arm"], r["scene_id"], r["case_id"], r["condition"]) for r in rows]
    if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
        raise RuntimeError("duplicate or unexpected episode")
    groups = []
    for condition in protocol["conditions"]:
        for arm in protocol["arms"]:
            cohort = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
            if not cohort:
                continue
            pipeline = [ms for r in cohort for ms in r["pipeline_step_ms"]]
            groups.append(dict(condition=condition, arm=arm, n=len(cohort),
                success=sum(bool(r["success_without_overlap"]) and not r["collision_union"] for r in cohort),
                collision=sum(bool(r["collision_union"]) for r in cohort),
                timeout=sum(bool(r["timeout"]) and not r["collision_union"] for r in cohort),
                penalty_time=float(np.mean([r["audited_time"] for r in cohort])),
                pipeline_p95_ms=float(np.percentile(pipeline, 95)),
                wall_seconds=sum(r["wall_seconds"] for r in cohort)))
    result = dict(complete=set(keys) == expected, completed=len(rows), expected=len(expected), groups=groups,
        limitations=["Exploratory mixed-density development, not an independent test or strict five-person generalization.",
            f"{protocol['particles']} particles; finite-ensemble convergence must be established before formal evaluation.",
            "C/D use the same filter design, but closed-loop observation histories diverge after different actions.",
            "B/C also differ in filtering approximation; do not attribute their entire gap solely to negative information.",
            "E hard-streak parameter is declared, not a fully tuned final baseline.",
            "The first compiled invocation includes JIT time; these timings are not a warmed latency study.",
            "The grid is an ideal simulated occupancy observation. Negative spatial evidence consumes only cells reported free; occupied and unknown cells never disprove a track.",
            "The severe condition corrupts identity-associated detections but not the occupancy grid; it is not a fully noisy range-sensor experiment."])
    write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-scene", type=int, default=2)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--backend", choices=("numba", "torch_cuda"), default="numba")
    parser.add_argument("--hard-streak", type=int, default=2)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.cases_per_scene < 1 or args.hard_streak < 1 or args.subdivisions < 2:
        raise ValueError("positive case count/hard-streak and at least two subdivisions required")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    cfg = unicycle_config(config(4))
    protocol = dict(arms=args.arms, scenes=SCENES, cases_per_scene=args.cases_per_scene,
        conditions=args.conditions, particles=args.particles, hard_streak=args.hard_streak,
        subdivisions=args.subdivisions,
        backend=args.backend,
        config=asdict(cfg), source=source_manifest(), worker_count=1,
        numba_threads=int(os.environ["NUMBA_NUM_THREADS"]),
        label="development-only reference prototype; previously revealed layouts",
        sensor_contract="ideal simulated occupancy grid plus identity-associated detections; negative spatial evidence consumes only observed-free cells; severe noise corrupts detections, not the grid")
    protocol = json.loads(json.dumps(protocol))
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if not args.summary_only and previous != protocol:
            raise RuntimeError("cannot resume with changed code or protocol")
        protocol = previous
    elif not args.summary_only:
        write_json(protocol_path, protocol)
        snapshot = out / "source"
        snapshot.mkdir(exist_ok=True)
        for name in SOURCE_FILES:
            (snapshot / name).write_bytes((ROOT / name).read_bytes())
    if args.summary_only:
        print(json.dumps(summarize(out, protocol), indent=2), flush=True)
        return
    for condition in args.conditions:
        for scene in SCENES:
            for case in range(scene[-1], scene[-1] + args.cases_per_scene):
                for arm in args.arms:
                    target = out / "episodes" / f"{condition}_{scene[0]}_{case}_{arm}.json"
                    if target.exists():
                        continue
                    if source_manifest() != protocol["source"]:
                        raise RuntimeError("source changed during experiment")
                    row = run_one(arm, scene, case, CONDITIONS[condition], cfg,
                                  args.particles, args.hard_streak, args.subdivisions,
                                  args.backend)
                    row["condition"] = condition
                    write_json(target, row)
                    print(json.dumps({k: row[k] for k in ("arm", "scene_id", "case_id", "condition", "event", "nav_time", "wall_seconds")}), flush=True)
                    summarize(out, protocol)
    final = summarize(out, protocol)
    if not final["complete"]:
        raise RuntimeError("incomplete prototype cohort")
    print("SPATIAL_PROTOTYPE_DONE", flush=True)


if __name__ == "__main__":
    main()
