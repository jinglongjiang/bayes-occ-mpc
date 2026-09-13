"""Frozen development attribution with a checked continuous executor."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

PROJECT = Path("/home/abc/workspace/bayes_occ_mpc")
HERE = Path(__file__).resolve().parent
SOURCE = HERE if (HERE / "continuous_mpc_gate.py").exists() else PROJECT
sys.path.insert(0, str(SOURCE))

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import binomtest

from continuous_mpc_gate import DEFAULT_CROWDNAV, ObservationAdapter, build_env, run_episode
from evaluate_matched_safety import config as point_config
from modern_dynamics import ContinuousExecutor, ContinuousUnicycleMPC
from unicycle_mpc_gate import unicycle_config

ARMS = ("current", "single_mode", "brake")
SCENES = (("circle5", 5, "circle_crossing", 4., None, 20100),
          ("square20", 20, "square_crossing", None, 10., 21000))
COUNT = 100
MODEL = "continuous_unicycle"
LOCAL_FILES = ("continuous_mpc_gate.py", "bayesian_rfs.py", "modern_dynamics.py",
               "unicycle_mpc_gate.py", "evaluate_matched_safety.py")


def cfg():
    return unicycle_config(point_config(4), horizon=16)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


class CheckedPlanner(ContinuousUnicycleMPC):
    def _cost(self, controls, *args):
        self.evaluations += len(controls)
        return super()._cost(controls, *args)

    def _degraded_choice(self, *args):
        self.fallback_calls += 1
        return super()._degraded_choice(*args)

    def plan(self, obs, seed):
        self.evaluations = self.fallback_calls = 0
        action, elapsed = super().plan(obs, seed)
        assert self.evaluations == self.cfg.population * self.cfg.iterations
        _, _, positions = ContinuousUnicycleMPC._rollout(self, self.last_controls[None], obs)
        self.predicted_next = positions[0, 0].copy()
        return action, elapsed


class SingleMode(CheckedPlanner):
    def _route_seeds(self, obs):
        return np.empty((0, self.cfg.horizon, 2))

    def _seed_trajectories(self, obs):
        # Keep the route candidates, but do not fit a separate distribution to each.
        return np.concatenate((super()._seed_trajectories(obs),
                               ContinuousUnicycleMPC._route_seeds(self, obs)))


class Brake(CheckedPlanner):
    def _degraded_choice(self, costs, first_physical, params, obs):
        self.fallback_calls += 1
        stop = self._seed_trajectories(obs)[1]
        expected_speed = max(self._speed(obs) - self.cfg.a_max * self.cfg.dt, 0.)
        np.testing.assert_allclose(stop[0], [expected_speed, 0.], atol=1e-12, rtol=0.)
        np.testing.assert_array_equal(params[1], stop)
        return 1


PLANNERS = dict(current=CheckedPlanner, single_mode=SingleMode, brake=Brake)


class CheckedExecutor(ContinuousExecutor):
    def __init__(self, config, planner_box):
        super().__init__(config)
        self.planner_box = planner_box
        self.records = []

    def __call__(self, env, command):
        assert abs(env.time_step - self.cfg.dt) < 1e-12
        planner = self.planner_box[0]
        speed_before = float(np.hypot(env.robot.vx, env.robot.vy))
        result = super().__call__(env, command)
        info = result[-1]
        assert info["execution_model"] == MODEL
        actual = np.array(env.robot.get_position())
        error = float(np.max(np.abs(actual - planner.predicted_next)))
        assert error < 1e-10, (actual, planner.predicted_next, error)
        assert info["acceleration_clip"] < 1e-10
        if isinstance(planner, Brake) and planner.last_diagnostics["feasibility_class"] == 2:
            np.testing.assert_allclose(
                [command.v, command.r],
                [max(speed_before - self.cfg.a_max * self.cfg.dt, 0.), 0.],
                atol=1e-12, rtol=0.)
        self.records.append(dict(execution_model=MODEL, prediction_error=error,
                                 executed_speed=info["executed_speed"],
                                 executed_turn=info["executed_turn"]))
        return result


def run_one(task):
    out, arm, scene, case, protocol_hash = task
    target = Path(out) / "episodes" / f"{scene[0]}_{case}_{arm}.json"
    if target.exists():
        row = json.loads(target.read_text())
        validate_row(row, arm, scene, case, protocol_hash)
        return row["wall_seconds"], str(target), True
    configuration = cfg()
    planner_box, frames, initial_layout = [], [], []

    def planner_factory(config):
        planner = PLANNERS[arm](config)
        planner_box.append(planner)
        return planner

    executor = CheckedExecutor(configuration, planner_box)

    def observe(env, obs, adapter, step):
        planner = planner_box[0]
        if step == 0:
            fields = ("px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta")
            layout = dict(robot=[getattr(env.robot, key) for key in fields],
                          humans=[[getattr(h, key) for key in fields] for h in env.humans])
            initial_layout.append(digest(layout))
        frames.append(dict(observation_hash=obs.provenance, evaluations=planner.evaluations,
                           fallback_calls=planner.fallback_calls,
                           before=[*obs.robot_xy, *obs.robot_velocity, obs.robot_heading],
                           predicted_next=planner.predicted_next.tolist()))

    start = time.monotonic()
    result = run_episode(
        DEFAULT_CROWDNAV, "bayes", scene[1], scene[2], case, configuration,
        scene[3], scene[4], 0, 0., 0., 1., time_limit=25,
        planner_type=planner_factory, robot_kinematics="unicycle", occlusion=True,
        step_executor=executor, step_observer=observe)
    row = asdict(result)
    assert len(row["steps"]) == len(frames) == len(executor.records)
    for step, frame, execution in zip(row["steps"], frames, executor.records):
        step.update(frame)
        step.update(execution)
    row.update(arm=arm, scene_id=scene[0], protocol_hash=protocol_hash,
               execution_model=MODEL, layout_hash=initial_layout[0],
               wall_seconds=time.monotonic() - start, development_only=True)
    success = bool(row["success_without_overlap"] and not row["collision_union"])
    row["audited_time"] = row["nav_time"] if success else 25.
    row["audited_event"] = ("collision" if row["collision_union"] else
                            "reach_goal" if success else "timeout")
    row["fallback_steps"] = sum(s["feasibility_class"] == 2 for s in row["steps"])
    validate_row(row, arm, scene, case, protocol_hash)
    save(target, row)
    return row["wall_seconds"], str(target), False


def validate_row(row, arm, scene, case, protocol_hash):
    assert (row["arm"], row["scene_id"], row["case_id"]) == (arm, scene[0], case)
    assert row["execution_model"] == MODEL
    assert row["protocol_hash"] == protocol_hash
    assert row["human_num"] == scene[1]
    assert row["solver_steps"] == len(row["steps"]) == len(row["plan_step_ms"])
    assert row["bound_violations"] == 0
    for step in row["steps"]:
        assert step["execution_model"] == MODEL
        assert step["prediction_error"] < 1e-10
        assert step["evaluations"] == 512 * 4
    assert np.isfinite(row["plan_step_ms"]).all()
    assert row["audited_event"] in ("reach_goal", "collision", "timeout")


def check_prefix(a, b):
    assert a["layout_hash"] == b["layout_hash"]
    fields = ("observation_hash", "before", "action_a", "action_b", "x", "y",
              "theta", "speed", "clearance", "feasibility_class")
    compared = 0
    for left, right in zip(a["steps"], b["steps"]):
        lf = left["feasibility_class"] == 2
        rf = right["feasibility_class"] == 2
        if lf or rf:
            assert lf and rf, "different first fallback trigger"
            assert left["observation_hash"] == right["observation_hash"]
            np.testing.assert_array_equal(left["before"], right["before"])
            return dict(prefix_steps=compared, first_fallback_index=compared)
        for field in fields:
            assert left[field] == right[field], (field, left[field], right[field])
        compared += 1
    assert len(a["steps"]) == len(b["steps"]), "diverged without fallback"
    assert a["audited_event"] == b["audited_event"]
    return dict(prefix_steps=compared, first_fallback_index=None)


def execution_acceptance():
    from continuous_mpc_gate import _load_modules
    _, _, _, _, ActionRot = _load_modules(DEFAULT_CROWDNAV)
    config = cfg()
    rows = []
    for speed0, speed1, turn, heading in (
        (0., .5, 0., 0.), (1., .5, 0., .4), (.4, 0., 0., 2.),
        (.2, .7, .2, .8), (.9, .4, -.2, 5.9), (.8, .8, .16, 1.2)):
        env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", 5, "circle_crossing", 4., None, 25, True)
        env.reset(options={"test_case": 20100})
        env.humans = []
        env.robot.kinematics = "unicycle"
        env.robot.theta = heading
        env.robot.vx, env.robot.vy = speed0 * np.cos(heading), speed0 * np.sin(heading)
        adapter = ObservationAdapter("bayes", config.horizon, config.human_margin, config.dt)
        obs = adapter.read(env)
        planner = ContinuousUnicycleMPC(config)
        params = np.tile([speed1, turn], (1, config.horizon, 1))
        projected, _, predicted = planner._rollout(params, obs)
        before = np.array(env.robot.get_position())
        result = ContinuousExecutor(config)(env, ActionRot(*projected[0, 0]))
        actual = np.array(env.robot.get_position())
        reference = np.array([
            quad(lambda t: (speed0 + (speed1-speed0)*t/config.dt) *
                 trig(heading + turn*t/config.dt), 0., config.dt,
                 epsabs=1e-13, epsrel=1e-13)[0]
            for trig in (np.cos, np.sin)])
        np.testing.assert_allclose(actual, predicted[0, 0], atol=1e-10, rtol=0.)
        np.testing.assert_allclose(actual-before, reference, atol=1e-10, rtol=0.)
        assert result[-1]["execution_model"] == MODEL
        rows.append(dict(speed_start=speed0, speed_end=speed1, turn=turn,
                         error=float(np.max(np.abs(actual-predicted[0, 0]))),
                         independent_integral_error=float(np.max(np.abs(actual-before-reference)))))
    return rows


def assets():
    paths = [SOURCE / name for name in LOCAL_FILES]
    paths.append(Path(__file__).resolve())
    paths.append(HERE / "test_mechanism_continuous.py")
    for folder in ("crowd_sim", "crowd_nav/configs"):
        paths.extend(p for p in (DEFAULT_CROWDNAV / folder).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".config"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def freeze(out):
    source = out / "source"
    source.mkdir(parents=True, exist_ok=False)
    for name in LOCAL_FILES:
        shutil.copy2(PROJECT / name, source / name)
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    shutil.copy2(HERE / "test_mechanism_continuous.py", source / "test_mechanism_continuous.py")
    os.execv(sys.executable, [sys.executable, "-u", str(source / Path(__file__).name),
                            "--out", str(out), "--frozen"])


def validate_all(out, protocol_hash):
    expected = {(arm, scene[0], case) for arm in ARMS for scene in SCENES
                for case in range(scene[-1], scene[-1] + COUNT)}
    rows = {}
    for path in (out / "episodes").glob("*.json"):
        row = json.loads(path.read_text())
        key = row["arm"], row["scene_id"], row["case_id"]
        assert key in expected and key not in rows
        scene = next(s for s in SCENES if s[0] == row["scene_id"])
        validate_row(row, key[0], scene, key[2], protocol_hash)
        rows[key] = row
    assert set(rows) == expected, (len(rows), len(expected))
    prefixes = []
    for scene in SCENES:
        hashes = set()
        for case in range(scene[-1], scene[-1] + COUNT):
            triplet = [rows[arm, scene[0], case] for arm in ARMS]
            assert len({r["layout_hash"] for r in triplet}) == 1
            assert triplet[0]["layout_hash"] not in hashes
            hashes.add(triplet[0]["layout_hash"])
            prefixes.append(dict(scene=scene[0], case=case, **check_prefix(triplet[0], triplet[2])))
    return rows, prefixes


def pairs(left, right, rng):
    events = ("reach_goal", "collision", "timeout")
    matrix = {a: {b: 0 for b in events} for a in events}
    changes, metrics = [], {}
    for a, b in zip(left, right):
        assert (a["scene_id"], a["case_id"]) == (b["scene_id"], b["case_id"])
        matrix[a["audited_event"]][b["audited_event"]] += 1
        if a["audited_event"] != b["audited_event"]:
            changes.append(dict(case=a["case_id"], current=a["audited_event"],
                                variant=b["audited_event"],
                                current_time=a["audited_time"], variant_time=b["audited_time"]))
    resamples = rng.integers(len(left), size=(20000, len(left)))
    for event in events:
        a = np.array([r["audited_event"] == event for r in left], dtype=int)
        b = np.array([r["audited_event"] == event for r in right], dtype=int)
        only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((b == 1) & (a == 0)).sum())
        ci = np.quantile((b-a)[resamples].mean(axis=1), [.025, .975])
        metrics[event] = dict(current_only=only_a, variant_only=only_b,
                              variant_minus_current=float((b-a).mean()),
                              bootstrap_ci95=ci.tolist(),
                              mcnemar_p=float(binomtest(only_a, only_a+only_b).pvalue)
                              if only_a+only_b else 1.)
    dt = np.array([b["audited_time"]-a["audited_time"] for a, b in zip(left, right)])
    metrics["time"] = dict(variant_minus_current=float(dt.mean()),
                           bootstrap_ci95=np.quantile(dt[resamples].mean(axis=1), [.025, .975]).tolist())
    return dict(n=len(left), event_transitions=matrix, changed_cases=changes, metrics=metrics)


def analyze(out, protocol_hash):
    rows, prefixes = validate_all(out, protocol_hash)
    table, paired = [], {}
    rng = np.random.default_rng(91827)
    for scene in SCENES:
        groups = {}
        for arm in ARMS:
            group = [rows[arm, scene[0], case] for case in range(scene[-1], scene[-1]+COUNT)]
            groups[arm] = group
            step_times = np.concatenate([r["plan_step_ms"] for r in group])
            table.append(dict(scene=scene[0], arm=arm, n=len(group),
                sr=np.mean([r["audited_event"] == "reach_goal" for r in group]),
                cr=np.mean([r["audited_event"] == "collision" for r in group]),
                tr=np.mean([r["audited_event"] == "timeout" for r in group]),
                audited_time=np.mean([r["audited_time"] for r in group]),
                plan_p95_ms=np.percentile(step_times, 95),
                fallback_episodes=sum(r["fallback_steps"] > 0 for r in group),
                fallback_steps=sum(r["fallback_steps"] for r in group),
                total_steps=sum(r["solver_steps"] for r in group)))
        for variant in ARMS[1:]:
            paired[f"{scene[0]}__current_vs_{variant}"] = pairs(groups["current"], groups[variant], rng)
    tests = [(name, event, pair["metrics"][event]["mcnemar_p"])
             for name, pair in paired.items() for event in ("reach_goal", "collision", "timeout")]
    last = 0.
    for i, (name, event, p) in enumerate(sorted(tests, key=lambda x: x[2])):
        last = min(1., max(last, p*(len(tests)-i)))
        paired[name]["metrics"][event]["holm_p_12_binary_comparisons"] = last
    result = dict(table=table, paired=paired, prefix_checks=prefixes,
                  timing_scope="2 CPU workers, 2 BLAS threads each; concurrent throughput measurement",
                  inference="development evidence; lack of significance is not equivalence",
                  bootstrap="20000 paired layout resamples within each scene; pointwise CI")
    save(out / "analysis.json", result)
    lines = ["# Corrected continuous-executor attribution", "",
             "600 development episodes; fixed Gaussian belief and point 4. Timing is measured under two-worker concurrency.", "",
             "| Scene | Arm | SR | audited CR | TR | penalized time s | plan p95 ms | fallback episodes | fallback steps / total |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['scene']} | {row['arm']} | {row['sr']:.1%} | {row['cr']:.1%} | {row['tr']:.1%} | {row['audited_time']:.3f} | {row['plan_p95_ms']:.2f} | {row['fallback_episodes']} | {row['fallback_steps']}/{row['total_steps']} |")
    for name, data in paired.items():
        lines += ["", f"## {name}", "", "Rows=current, columns=variant.", "",
                  "| Event | reach_goal | collision | timeout |", "|---|---:|---:|---:|"]
        for event, values in data["event_transitions"].items():
            lines.append(f"| {event} | {values['reach_goal']} | {values['collision']} | {values['timeout']} |")
        lines += ["", "```json", json.dumps(data["metrics"], indent=2), "```", "",
                  "All changed cases: " + json.dumps(data["changed_cases"])]
    lines += ["", f"Prefix agreement passed for all {len(prefixes)} current/brake pairs.",
              "No equivalence or noninferiority claim is inferred from nonsignificance."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.frozen:
        freeze(out)
    manifest = assets()
    protocol = dict(arms=ARMS, scenes=SCENES, cases_per_scene=COUNT, config=asdict(cfg()),
        noise=[0., 0., 1.], occlusion=True, robot_visible=False, time_limit=25,
        planner_seed_offset=0, execution_model=MODEL, workers=2, blas_threads=2,
        python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
        host=platform.node(), source_manifest=manifest,
        single_mode="one fitted Gaussian; seven standard seeds and three route candidates retained",
        brake="projected stop seed index 1, only selected if best feasibility class is 2",
        scope="fixed development cases; partly reused from earlier diagnostics, not held-out confirmation",
        decisions="retain current on uncertainty; nonsignificance does not authorize removal",
        previous_results="runs/mechanism used mismatched execution and cannot establish continuous-model attribution")
    protocol_hash = digest(protocol)
    if (out / "protocol.json").exists():
        assert json.loads((out / "protocol.json").read_text()) == json_safe(protocol)
    else:
        save(out / "protocol.json", protocol)
    if args.analyze_only:
        analyze(out, protocol_hash)
        return
    started = time.monotonic()
    acceptance = execution_acceptance()
    save(out / "acceptance.json", dict(single_step=acceptance, status="PASSED"))
    print("EXECUTION_ACCEPTANCE_PASSED", json.dumps(acceptance), flush=True)
    # The first paired runs become part of the registered 600, not extra samples.
    for scene in SCENES:
        for arm in ("current", "brake"):
            elapsed, path, _ = run_one((str(out), arm, scene, scene[-1], protocol_hash))
            print(f"PREFLIGHT {arm} {scene[0]} wall={elapsed:.1f}s {path}", flush=True)
        a = json.loads((out / "episodes" / f"{scene[0]}_{scene[-1]}_current.json").read_text())
        b = json.loads((out / "episodes" / f"{scene[0]}_{scene[-1]}_brake.json").read_text())
        acceptance_row = check_prefix(a, b)
        print("PREFIX_ACCEPTANCE_PASSED", scene[0], acceptance_row, flush=True)
    tasks = [(str(out), arm, scene, case, protocol_hash)
             for case_offset in range(COUNT) for scene in SCENES for arm in ARMS
             for case in (scene[-1]+case_offset,)
             if not (out / "episodes" / f"{scene[0]}_{case}_{arm}.json").exists()]
    completed = 600 - len(tasks)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                elapsed, path, cached = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            completed += 1
            if completed % 6 == 0 or completed == 600:
                runtime = time.monotonic() - started
                print(f"PROGRESS {completed}/600 wall={runtime:.1f}s last_episode={elapsed:.1f}s", flush=True)
                save(out / "progress.json", dict(completed=completed, expected=600,
                     elapsed_seconds=runtime, last_episode_seconds=elapsed, last_file=path))
    assert assets() == manifest, "source changed during experiment"
    result = analyze(out, protocol_hash)
    save(out / "complete.json", dict(status="COMPLETE", episodes=600,
         elapsed_seconds=time.monotonic()-started, protocol_hash=protocol_hash,
         prefix_pairs=len(result["prefix_checks"])))
    print("COMPLETE", json.dumps(json_safe(result["table"])), flush=True)


if __name__ == "__main__":
    main()
