#!/usr/bin/env python3
"""Experiment-only covariance interventions; the production planner has one path."""

import argparse
import concurrent.futures as futures
from dataclasses import asdict, replace
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from continuous_mpc_gate import (
    ContinuousCEMMPC, MPCConfig, ObservationAdapter, DEFAULT_CROWDNAV,
    run_episode, verify_pairing, verify_access_contract,
)
from bayesian_rfs import RFSConfig

ROOT = Path(__file__).resolve().parent
if (ROOT / "environment").is_dir():
    DEFAULT_CROWDNAV = ROOT / "environment"
SCENES = [
    ("baseline_circle", 5, "circle_crossing", 4., None),
    ("baseline_square", 10, "square_crossing", None, 10.),
    ("dense_circle", 10, "circle_crossing", 4., None),
    ("dense_square", 20, "square_crossing", None, 10.),
    ("large_circle", 12, "circle_crossing", 6., None),
    ("large_square", 20, "square_crossing", None, 14.),
]
RISK_SCALES = (.5, .75, 1., 1.5, 2.)
METHODS = ("bayes", "static_cov", "ewma", "conformal", "fixed")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def config(point=2):
    scale = RISK_SCALES[point]
    return MPCConfig(population=512, iterations=4, v_max=1.,
                     chance_limit=min(.9, .5 * scale), near_chance_limit=.15 * scale,
                     probability_weight=1., human_margin=.10)


class MatchedAdapter(ObservationAdapter):
    """Keep the same filter, track means, existence and integrator in all Gaussian arms."""

    def __init__(self, *args, method="bayes", calibration=None, point=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.calibration = calibration
        self.point = point
        self.pending = {}
        self.scales = {}

    def read(self, env):
        observation = super().read(env)
        step = int(round(env.global_time / self.rfs.cfg.dt))
        detections = {int(e["id"]): np.array([e["px"], e["py"]])
                      for e in self.detected_entities}
        if self.calibration is not None:
            base = np.asarray(self.calibration["variance_by_visibility_horizon"])
            alpha = self.calibration["ewma_alpha"]
            # Multiple past horizons can mature now; use only the most recent
            # forecast for each detected object, so one measurement counts once.
            matured = self.pending.pop(step, {})
            for identifier, (horizon, mean, variance) in matured.items():
                if identifier in detections:
                    residual = detections[identifier] - mean
                    relative_variance = float(residual @ residual / (2. * variance))
                    old = self.scales.get(identifier, 1.)
                    self.scales[identifier] = max(1e-6, (1. - alpha) * old + alpha * relative_variance)
            for i, identifier in enumerate(self.reported_ids):
                track = self.rfs.tracks[identifier]
                visible = int(track.visible)
                for h in range(self.horizon):
                    target = self.pending.setdefault(step + h + 1, {})
                    entry = (h + 1, observation.human_segment_end[i, h].copy(), base[visible, h])
                    if identifier not in target or h + 1 < target[identifier][0]:
                        target[identifier] = entry
            visible = np.array([int(self.rfs.tracks[i].visible) for i in self.reported_ids], dtype=int)
            if self.method in ("static_cov", "ewma"):
                variance = base[visible].copy()
                if self.method == "static_cov":
                    variance *= self.calibration["static_scale"]
                else:
                    variance *= np.array([self.scales.get(i, 1.) for i in self.reported_ids])[:, None]
                observation.human_position_covariance = variance[..., None, None] * np.eye(2)
            elif self.method in ("conformal", "fixed"):
                if self.method == "conformal":
                    radii = np.asarray(self.calibration["conformal_radii"][str(self.point)])
                    observation.human_uncertainty_buffer = radii[visible]
                else:
                    observation.human_uncertainty_buffer = np.full(
                        (len(visible), self.horizon), self.calibration["fixed_radii"][self.point])
                observation.human_position_covariance = None
                observation.human_existence = None
        if self.method == "posterior_mean":
            observation.human_position_covariance = None
            observation.human_existence = None
        payload = [observation.provenance, self.method]
        for value in (observation.human_position_covariance, observation.human_uncertainty_buffer):
            if value is not None:
                payload.append(hashlib.sha256(value.tobytes()).hexdigest())
        observation.provenance = ":".join(payload)
        return observation


class CVMemoryAdapter(ObservationAdapter):
    """Last detection plus constant velocity, without probabilistic filtering."""

    def __init__(self, *args, max_age=2.0, **kwargs):
        args = ("sensor",) + args[1:]
        super().__init__(*args, **kwargs)
        self.memory = {}
        self.max_age = float(max_age)

    def read(self, env):
        observation = super().read(env)
        timestamp = float(env.global_time)
        detected = {int(e['id']) for e in self.detected_entities}
        for entity in self.detected_entities:
            self.memory[int(entity['id'])] = (dict(entity), timestamp)
        rows, identifiers = [], []
        mx, my = self.sensor_mesh
        resolution = float(mx[0, 1] - mx[0, 0])
        for identifier, (entity, observed_at) in list(self.memory.items()):
            age = timestamp - observed_at
            xy = np.array([entity['px'], entity['py']]) + age * np.array([entity['vx'], entity['vy']])
            col = int(np.floor((xy[0] - mx[0, 0]) / resolution + .5))
            row = int(np.floor((xy[1] - my[0, 0]) / resolution + .5))
            visible_empty = (0 <= row < my.shape[0] and 0 <= col < mx.shape[1]
                             and self.sensor_grid[row, col] == 0.)
            if age > self.max_age or (identifier not in detected and visible_empty):
                del self.memory[identifier]
                continue
            rows.append([*xy, entity['vx'], entity['vy'], entity['radius']])
            identifiers.append(identifier)
        observation.entities = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
        starts = np.arange(self.horizon) * self.dt
        ends = starts + self.dt
        xy = observation.entities[:, None, :2]
        velocity = observation.entities[:, None, 2:4]
        observation.human_segment_start = xy + velocity * starts[None, :, None]
        observation.human_segment_end = xy + velocity * ends[None, :, None]
        self.reported_ids = identifiers
        observation.provenance += ':cv_memory:' + hashlib.sha256(observation.entities.tobytes()).hexdigest()
        return observation


def planner_class(search):
    if search == "new":
        return ContinuousCEMMPC
    # Only the plan() sampling/elite update is inherited. Every dynamics, cost,
    # numerical probability and feasibility function is the current implementation.
    path = ROOT / "legacy_search.py"
    if not path.exists():
        path = ROOT / "results/pre_route_upgrade/source/continuous_mpc_gate.py"
    spec = importlib.util.spec_from_file_location("archived_search", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    class SingleDistributionSearch(ContinuousCEMMPC):
        plan = module.ContinuousCEMMPC.plan
    return SingleDistributionSearch


def evaluate_one(spec):
    scene, case, method, point, calibration, search, noise, seed = spec
    _, count, sim, radius, width = scene
    cfg = config(point)
    if method == "fixed":
        cfg = config(2)
    return asdict(run_episode(DEFAULT_CROWDNAV, "bayes", count, sim, case, cfg,
        radius, width, seed, *noise, time_limit=25,
        planner_type=planner_class(search),
        adapter_factory=partial(MatchedAdapter, method=method, calibration=calibration, point=point)))


class CalibrationRecorder:
    def __init__(self):
        self.pending = {}
        self.rows = []

    def __call__(self, env, observation, adapter, step):
        # Called only AFTER control selection. Truth is offline supervision, never
        # passed back to the observation adapter or planner.
        detected = {int(e["id"]): np.array([e["px"], e["py"]]) for e in adapter.detected_entities}
        for identifier, origin, h, visible, mean in self.pending.pop(step, []):
            truth = np.asarray(env.humans[identifier].get_position())
            error = truth - mean
            measured_error = detected[identifier] - mean if identifier in detected else np.zeros(2)
            self.rows.append([identifier, origin, h, visible, *error,
                              int(identifier in detected), *measured_error])
        for i, identifier in enumerate(adapter.reported_ids):
            for h in range(adapter.horizon):
                self.pending.setdefault(step + h + 1, []).append((identifier, step, h + 1,
                    int(adapter.rfs.tracks[identifier].visible), observation.human_segment_end[i, h].copy()))


def calibrate_one(task):
    case, destination = task
    recorder = CalibrationRecorder()
    result = run_episode(DEFAULT_CROWDNAV, "bayes", 5, "circle_crossing", case,
        config(), circle_radius=4., time_limit=25, step_observer=recorder)
    path = Path(destination) / f"case_{case}.npz"
    np.savez_compressed(path, rows=np.asarray(recorder.rows, dtype=float).reshape(-1, 9))
    return asdict(result)


def ewma_validation(rows, base, alpha):
    updates = {}
    for row in rows:
        identifier, origin, h, visible = row[:4].astype(int)
        if row[6]:
            key = (origin + h, identifier)
            if key not in updates or h < updates[key][0]:
                updates[key] = (h, float(row[7:9] @ row[7:9] / (2 * base[visible, h-1])))
    states, total, n = {}, 0., 0
    for step in np.unique(rows[:, 1]).astype(int):
        for (target, identifier), (_, residual) in updates.items():
            if target == step:
                states[identifier] = max(1e-6, (1-alpha) * states.get(identifier, 1.) + alpha * residual)
        batch = rows[rows[:, 1] == step]
        ids, h, visible = batch[:, 0].astype(int), batch[:, 2].astype(int), batch[:, 3].astype(int)
        variance = base[visible, h-1] * np.asarray([states.get(i, 1.) for i in ids])
        error2 = np.sum(batch[:, 4:6] ** 2, axis=1)
        total += float(np.sum(np.log(2*np.pi*variance) + error2/(2*variance)))
        n += len(batch)
    return total, n


def fit_calibration(args):
    directory = args.output.parent / "calibration_records"
    directory.mkdir(parents=True)
    cases = list(range(args.offset, args.offset + args.count))
    with futures.ProcessPoolExecutor(args.workers) as executor:
        for r in executor.map(calibrate_one, [(case, str(directory)) for case in cases]):
            print("calibration", r["case_id"], r["event"], flush=True)
    midpoint = len(cases)//2
    fit = np.concatenate([np.load(directory/f"case_{c}.npz")["rows"] for c in cases[:midpoint]])
    validation = [np.load(directory/f"case_{c}.npz")["rows"] for c in cases[midpoint:]]
    base = np.zeros((2, config().horizon))
    errors = {}
    for visible in (0,1):
        for h in range(1,config().horizon+1):
            e = fit[(fit[:,3]==visible)&(fit[:,2]==h),4:6]
            if len(e) < 20:
                raise RuntimeError(f"insufficient calibration stratum {visible}/{h}: {len(e)}")
            errors[visible,h] = np.sort(np.linalg.norm(e,axis=1))
            base[visible,h-1] = max(1e-6,float(np.mean(e**2)))
    static_scores = {}
    for scale in (.5,.75,1.,1.5,2.):
        scores = []
        for rows in validation:
            v = base[rows[:,3].astype(int),rows[:,2].astype(int)-1]*scale
            scores.append(float(np.mean(np.log(2*np.pi*v)+np.sum(rows[:,4:6]**2,axis=1)/(2*v))))
        static_scores[str(scale)] = float(np.mean(scores))
    ewma_scores = {}
    for alpha in (.02,.05,.1,.2,.4):
        values = [ewma_validation(rows,base,alpha) for rows in validation]
        ewma_scores[str(alpha)] = float(np.mean([total/n for total,n in values]))
    radii = {}
    for point in range(len(RISK_SCALES)):
        limits = ContinuousCEMMPC(config(point))._belief_risk_limits()
        table = np.zeros_like(base)
        for visible in (0,1):
            for h, limit in enumerate(limits,1):
                e = errors[visible,h]
                rank = min(len(e),int(np.ceil((len(e)+1)*(1-limit))))
                table[visible,h-1] = float(e[max(0,rank-1)])
        radii[str(point)] = table.tolist()
    save(args.output,{"variance_by_visibility_horizon":base.tolist(),
        "static_scale":float(min(static_scores,key=static_scores.get)),
        "static_validation_nll":static_scores,
        "ewma_alpha":float(min(ewma_scores,key=ewma_scores.get)),
        "ewma_validation_nll":ewma_scores,"conformal_radii":radii,
        "fixed_radii":[.15,.25,.35,.45,.60],
        "cases_fit":cases[:midpoint],"cases_validation":cases[midpoint:],
        "human_count":5,"robot_visible":False,
        "conformal_scope":"Empirical radial sets; overlapping forecasts are not independent samples, no distribution-free trajectory guarantee claimed."})


def summarize(rows):
    binary = ("success","collision","timeout","collision_union","success_without_overlap")
    result = {key:float(np.mean([r[key] for r in rows])) for key in binary}
    audited_time = [r["nav_time"] if r["success_without_overlap"] else 25. for r in rows]
    result.update(n=len(rows),penalized_time=float(np.mean([r["scored_time"] for r in rows])),
        audited_penalized_time=float(np.mean(audited_time)),
        mean_plan_ms=float(np.mean(np.concatenate([r["plan_step_ms"] for r in rows]))),
        pooled_p95_pipeline_ms=float(np.percentile(np.concatenate([r["pipeline_step_ms"] for r in rows]),95)),
        deadline_miss_fraction=float(np.mean(np.concatenate([r["pipeline_step_ms"] for r in rows])>250)),
        benchmark_success_with_overlap=sum(r["success"] and r["actual_overlap_steps"]>0 for r in rows))
    return result


def evaluate(args):
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    scene = SCENES[args.scene]
    noise = (args.position_noise,args.velocity_noise,args.detection_probability)
    expected = list(range(args.offset,args.offset+args.count))
    pairing = verify_pairing(DEFAULT_CROWDNAV,scene[1],scene[2],scene[3],scene[4])
    access = verify_access_contract(DEFAULT_CROWDNAV,scene[1],scene[2],scene[3],scene[4])
    tasks = [(scene,case,args.method,args.point,calibration,args.search,noise,args.seed) for case in expected]
    rows = []
    start = time.time()
    with futures.ProcessPoolExecutor(args.workers) as executor:
        for row in executor.map(evaluate_one,tasks):
            rows.append(row)
            print(args.method,scene[0],row["case_id"],row["event"],"swept",row["actual_min_clearance"],flush=True)
            save(args.output.with_suffix(".progress.json"),{"completed":len(rows),"total":len(tasks),"elapsed_seconds":time.time()-start})
    if [r["case_id"] for r in rows] != expected:
        raise RuntimeError("episode alignment failed")
    save(args.output,{"protocol":{"method":args.method,"point":args.point,"search":args.search,
        "scene":scene,"robot_visible":False,"time_limit":25,"planner":asdict(config(2 if args.method=="fixed" else args.point)),
        "noise":noise,"seed":args.seed,"pairing":pairing,"access":access,
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("continuous_mpc_gate.py","bayesian_rfs.py","evaluate_matched_safety.py")},
        "calibration_sha256":hashlib.sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
        "audit":"Benchmark outcomes unchanged; collision_union includes actual swept overlap before benchmark termination. Not a counterfactual rerun past benchmark collisions.",
        "elapsed_seconds":time.time()-start},"episodes":rows,"summary":summarize(rows)})


def make_report(out):
    from analyze_paired import binary_pair, holm_adjust
    out = Path(out)
    files = sorted(out.glob("formal_*.json"))
    if not files:
        return
    groups = {}
    for path in files:
        if path.name.endswith(".progress.json"):
            continue
        data = json.loads(path.read_text())
        protocol = data["protocol"]
        key = f"{protocol['method']}_p{protocol['point']}_{protocol['search']}"
        groups.setdefault(key,{})[protocol["scene"][0]] = data
    complete = {key:value for key,value in groups.items() if len(value)==6}
    metrics = ("success","collision","timeout","collision_union","success_without_overlap","penalized_time","audited_penalized_time")
    macro = {key:{metric:float(np.mean([v[scene[0]]["summary"][metric] for scene in SCENES])) for metric in metrics} for key,v in complete.items()}

    def matrix(key,metric):
        value = []
        expected = None
        for scene in SCENES:
            rows = sorted(complete[key][scene[0]]["episodes"],key=lambda r:r["case_id"])
            cases = [r["case_id"] for r in rows]
            if expected is None: expected = cases
            if cases != expected: raise RuntimeError("unpaired scenes")
            if metric == "audited_time":
                value.append([r["nav_time"] if r["success_without_overlap"] else 25. for r in rows])
            else: value.append([r[metric] for r in rows])
        return np.asarray(value)

    def interval(delta):
        delta = delta.mean(axis=0) if delta.ndim==2 else delta
        rng = np.random.default_rng(72615)
        samples = np.concatenate([delta[rng.integers(0,len(delta),(1000,len(delta)))].mean(axis=1) for _ in range(20)])
        return {"delta":float(np.mean(delta)),"ci95":np.quantile(samples,[.025,.975]).tolist(),
                "upper_bonferroni_8_endpoints":float(np.quantile(samples,1-.05/8))}

    contrasts = {}
    reference = "bayes_p2_new"
    if reference in complete:
        for key in complete:
            if key==reference: continue
            contrasts[key] = {metric:interval(matrix(reference,metric)-matrix(key,metric))
                for metric in ("success","collision","timeout","collision_union","audited_time")}
            paired = []
            for scene in SCENES:
                b = sorted(complete[reference][scene[0]]["episodes"],key=lambda r:r["case_id"])
                a = sorted(complete[key][scene[0]]["episodes"],key=lambda r:r["case_id"])
                if [r["case_id"] for r in b] != [r["case_id"] for r in a]: raise RuntimeError("case mismatch")
                differences = [x["nav_time"]-y["nav_time"] for x,y in zip(b,a)
                               if x["success_without_overlap"] and y["success_without_overlap"]]
                paired.append({"scene":scene[0],"n":len(differences),
                               "difference":interval(np.asarray(differences)) if differences else None})
            contrasts[key]["paired_successful_arrival"] = paired
    interaction = {}
    four = ["bayes_p2_new","static_cov_p2_new","bayes_p2_old","static_cov_p2_old"]
    if all(k in complete for k in four):
        for metric in ("success","collision_union","timeout","audited_time"):
            a,b,c,d = [matrix(k,metric) for k in four]
            interaction[metric] = interval((a-b)-(c-d))
    matched = {}
    selection_path = out / "frozen_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())["points"]
        bk = f"bayes_p{selection['bayes']}_new"
        if bk in complete:
            for method in METHODS[1:]:
                key = f"{method}_p{selection[method]}_new"
                if key not in complete: continue
                effects = {metric:interval(matrix(bk,metric)-matrix(key,metric))
                           for metric in ("collision_union","audited_time","success_without_overlap","timeout")}
                effects["supports_predeclared_safety_and_time"] = (
                    effects["collision_union"]["upper_bonferroni_8_endpoints"] < .01
                    and effects["audited_time"]["upper_bonferroni_8_endpoints"] < 0)
                matched[method] = effects
    supplementary, failures = {}, []
    for path in sorted(out.glob("*.json")):
        if path.stem.startswith(("noise_","seed_","latency_")) and not path.name.endswith(".progress.json"):
            data = json.loads(path.read_text())
            supplementary[path.stem] = data["summary"]
    for key, value in complete.items():
        if key not in four: continue
        for scene,data in value.items():
            for row in data["episodes"]:
                if not row["success_without_overlap"]:
                    failures.append({"method":key,"scene":scene,**{k:row[k] for k in
                        ("case_id","event","actual_min_clearance","actual_overlap_steps","first_actual_overlap_time","nav_time","min_clearance")}})
    pv = {}
    if reference in complete:
        for key in complete:
            if key==reference: continue
            for metric in ("success","collision_union","timeout"):
                # Scene-wise exact paired tests; macro intervals cluster all six
                # same-case layouts together instead of claiming 600 independent trials.
                b,a = matrix(reference,metric),matrix(key,metric)
                for i,scene in enumerate(SCENES):
                    pv[f"{key}:{scene[0]}:{metric}"] = binary_pair(b[i],a[i])["mcnemar_exact_p"]
    save(out/"analysis.json",{"macro":macro,"contrasts_reference_bayes_p2":contrasts,
        "search_covariance_interaction":interaction,"matched_dev_selected":matched,
        "supplementary":supplementary,"holm_scene_p":holm_adjust(pv),
        "ci_scope":"Case-cluster bootstrap; nominal 95% exploratory intervals. Main safety/time comparisons use a predeclared one-sided Bonferroni family of eight endpoints. Not a deployment safety certificate."})
    save(out/"failures.json",failures)
    lines = ["# Matched Safety and Attribution", "", "Frozen fresh-case evaluation. Benchmark and executed-motion collision checks are both retained.",
             "", "| Method / point / search | SR | benchmark CR | audited collision union | TR | audited penalized time |",
             "|---|---:|---:|---:|---:|---:|"]
    for key,values in macro.items():
        lines.append(f"| {key} | {values['success']:.2%} | {values['collision']:.2%} | {values['collision_union']:.2%} | {values['timeout']:.2%} | {values['audited_penalized_time']:.3f}s |")
    lines += ["", "## Dev-Selected Matched Comparisons", ""]
    for method,effects in matched.items():
        lines.append(f"- Bayes vs {method}: {json.dumps(effects)}")
    lines += ["", "## Attribution", json.dumps(interaction,indent=2), "", "## Supplementary",json.dumps(supplementary,indent=2),
        "", "## Limits", "Fixed/EWMA covariance arms share means, existence and the same Gaussian collision integral with Bayes. Conformal/fixed-radius are separate set-method controls, not one-factor covariance ablations.",
        "All five preregistered operating points are reported, not only winners. The searched grid is not the entire continuous Pareto frontier.",
        "Actual-motion audit does not simulate beyond a benchmark terminal collision. Neither simulator scores nor sub-250ms desktop planning establish real-robot deployment safety.",
        "Five-person fitting and fresh cases do not erase previous high-density algorithm-development exposure. SARL/LSTM remain historical references only.",
        "Statistical improvement is not by itself a novelty or publication verdict."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task",choices=("calibrate","evaluate"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--method",choices=METHODS+("posterior_mean",),default="bayes")
    parser.add_argument("--point",type=int,choices=range(5),default=2)
    parser.add_argument("--scene",type=int,choices=range(6),default=0)
    parser.add_argument("--search",choices=("new","old"),default="new")
    parser.add_argument("--offset",type=int,required=True)
    parser.add_argument("--count",type=int,default=100)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--position-noise",type=float,default=0.)
    parser.add_argument("--velocity-noise",type=float,default=0.)
    parser.add_argument("--detection-probability",type=float,default=1.)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.task == "calibrate": fit_calibration(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
