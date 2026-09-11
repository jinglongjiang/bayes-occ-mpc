"""Verify frozen episode records and summarize paired main-table experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import modern_main_table as run
from modern_repair import write_result


def key(record):
    t = record["task"]
    return (t["scene"][0], t["case"], t["seed"], t["occluded"])


def metrics(row):
    sr = int(row["success_without_overlap"])
    cr = int(row["collision_union"])
    return np.array([sr, cr, int(row["timeout"]) * (1-cr),
                     row["nav_time"] if sr else 25.], dtype=float)


def holm(values):
    order = np.argsort(values)
    result = np.zeros(len(values))
    maximum = 0.
    for i, index in enumerate(order):
        maximum = max(maximum, min(1., (len(values)-i)*values[index]))
        result[index] = maximum
    return result.tolist()


def block_sign_flip(block_sums):
    """Exact two-sided sign-flip probability for integer event differences."""
    integers = np.rint(block_sums).astype(int)
    np.testing.assert_allclose(integers, block_sums, atol=1e-10, rtol=0.)
    mass = np.ones(1)
    offset = 0
    for weight in np.abs(integers):
        if weight == 0:
            continue
        shifted = np.zeros(len(mass)+2*weight)
        shifted[:len(mass)] += .5*mass
        shifted[2*weight:] += .5*mass
        mass = shifted
        offset += weight
    values = np.arange(len(mass))-offset
    return min(1., float(mass[np.abs(values) >= abs(integers.sum())].sum()))


def paired(left, right, repetitions=10000):
    a, b = {key(r): r for r in left}, {key(r): r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("unpaired or duplicated layout/seed/condition")
    keys = sorted(a)
    av = np.asarray([metrics(a[k]["result"]) for k in keys])
    bv = np.asarray([metrics(b[k]["result"]) for k in keys])
    difference = av-bv
    # CrowdSim seeds NumPy from case_id, not from (scene, case_id). Keep all
    # scene variants and repeated planning seeds together in one resampling block.
    clusters = {}
    for i, k in enumerate(keys):
        clusters.setdefault((k[1], k[3]), []).append(i)
    cluster_totals = np.asarray([difference[v].sum(axis=0) for v in clusters.values()])
    cluster_sizes = np.asarray([len(v) for v in clusters.values()])
    rng = np.random.default_rng(91370)
    indices = rng.integers(0,len(clusters),size=(repetitions,len(clusters)))
    bootstrap = cluster_totals[indices].sum(axis=1) / cluster_sizes[indices].sum(axis=1)[:,None]
    events = {}
    for j, name in enumerate(("SR", "CR", "TR")):
        wins = int(np.sum((av[:, j] == 1) & (bv[:, j] == 0)))
        losses = int(np.sum((av[:, j] == 0) & (bv[:, j] == 1)))
        independent = len(clusters) == len(keys)
        block_sums = [difference[v,j].sum() for v in clusters.values()]
        events[name] = dict(left_only=wins, right_only=losses,
            p_block_sign_flip=block_sign_flip(block_sums),
            p_exact=binomtest(wins, wins+losses, .5).pvalue if wins+losses and independent
                    else 1. if independent else None)
    layouts = len({(k[0],k[1],k[3]) for k in keys})
    return dict(n=len(keys), layouts=layouts, seed_blocks=len(clusters),
                left_minus_right=difference.mean(axis=0).tolist(),
                paired_ci95=np.quantile(bootstrap, [.025, .975], axis=0).T.tolist(), events=events)


def summarize(records):
    rows = [r["result"] for r in records]
    successful = [r for r in rows if r["success_without_overlap"]]
    values = np.asarray([metrics(r) for r in rows])
    native = [s for r in rows for s in r.get("native_steps", [])]
    steps = np.concatenate([r["pipeline_step_ms"] for r in rows])
    plans = np.concatenate([r["plan_step_ms"] for r in rows])
    result = dict(n=len(rows), SR=values[:, 0].mean(), CR=values[:, 1].mean(),
        TR=values[:, 2].mean(), penalty_time=values[:, 3].mean(),
        successes=int(values[:, 0].sum()), collisions=int(values[:, 1].sum()),
        timeouts=int(values[:, 2].sum()),
        path_length=float(np.mean([r["path_length"] for r in rows])),
        success_only_time=float(np.mean([r["nav_time"] for r in successful])) if successful else None,
        success_only_path=float(np.mean([r["path_length"] for r in successful])) if successful else None,
        requested_action_saturations=sum(r["bound_violations"] for r in rows),
        max_requested_bound_excess=max(r["max_bound_excess"] for r in rows),
        steps=len(steps), pipeline_ms=dict(mean=float(steps.mean()), p50=float(np.median(steps)),
            p95=float(np.quantile(steps,.95)), p99=float(np.quantile(steps,.99)),
            deadline_miss=float(np.mean(steps > 250.))),
        plan_ms=dict(mean=float(plans.mean()), p95=float(np.quantile(plans,.95))))
    collisions = [r for r in rows if r["collision_union"]]
    failure = dict(collisions=len(collisions), collision_step_end_speed_below_01=sum(
        r["steps"][-1]["speed"] < .1 for r in collisions),
        note="post-hoc descriptions, not proof of a causal mechanism; speed is at the end of the collision interval, not necessarily at first contact")
    native_collisions = [r for r in collisions if r.get("native_steps")]
    if native_collisions:
        failure.update(
            last_command_solved=sum(bool(r["native_steps"][-1]["success"]) for r in native_collisions),
            failure_within_last_five_commands=sum(any(not s["success"] for s in r["native_steps"][-5:])
                                                 for r in native_collisions))
        valid_slack = [r["native_steps"][-1] for r in native_collisions
                       if r["native_steps"][-1]["success"] and
                       r["native_steps"][-1].get("slack1") is not None and
                       r["native_steps"][-1]["slack1"] >= 0.]
        if valid_slack:
            failure["solved_collision_steps_with_nonzero_slack"] = sum(s["slack1"] > 1e-3 for s in valid_slack)
            failure["solved_collision_steps_with_slack_record"] = len(valid_slack)
    result["posthoc_failure_descriptives"] = failure
    if native:
        result["native_solved_fraction"] = float(np.mean([s["success"] for s in native]))
        result["native_diagnostic_fields"] = sorted(set().union(*(set(s) for s in native)))
        for field in ("slack1", "scenario_validation_status", "qp_status", "acados_status"):
            solution_field = field in ("slack1", "scenario_validation_status")
            numbers = [s[field] for s in native if s.get(field) is not None
                       and (not solution_field or s["success"])]
            if numbers:
                result[field] = dict(mean=float(np.mean(numbers)), max=float(np.max(numbers)),
                                     nonzero=int(np.count_nonzero(numbers)), count=len(numbers),
                                     population="accepted_native_solutions" if solution_field else "all_solver_replies")
                if field.endswith("status"):
                    result[field]["counts"] = {str(int(v)): int(np.sum(np.asarray(numbers)==v))
                                                for v in sorted(set(numbers))}
                else:
                    result[field]["above_1e_minus3"] = int(np.sum(np.asarray(numbers) > 1e-3))
    result["per_scene"] = {}
    for scene in sorted({r["task"]["scene"][0] for r in records}):
        vals = np.asarray([metrics(r["result"]) for r in records if r["task"]["scene"][0] == scene])
        result["per_scene"][scene] = dict(zip(("SR", "CR", "TR", "penalty_time"), vals.mean(axis=0).tolist()))
    return result


def verify(records, frozen, expected_count, split):
    if len(records) != expected_count or len({r["task"]["key"] for r in records}) != len(records):
        raise ValueError(f"{split}: incomplete/duplicate records")
    selected = {c["id"]: c for c in frozen["selected"]}
    protocol = json.loads((run.OUT / "protocol.json").read_text())
    checked_manifests = set()
    for record in records:
        t, row = record["task"], record["result"]
        raw_task = {k: v for k, v in t.items() if k != "key"}
        if run.digest(raw_task) != t["key"] or t["split"] != split:
            raise ValueError("task identity changed")
        if t["candidate"] != selected[t["candidate"]["id"]]:
            raise ValueError("unfrozen candidate")
        name = t["scene"][0]
        registry_split = "T" if split == "S" else split
        if row["layout_sha256"] != protocol["layouts"][f"{registry_split}/{name}/{t['case']}"]:
            raise ValueError("layout mismatch")
        if row["case_id"] != t["case"] or row["occluded"] != t["occluded"]:
            raise ValueError("result paired with wrong task")
        if row["execution_model"] != "continuous_unicycle" or row["robot_visible"] or row["goal_radius"] != .3:
            raise ValueError("task contract changed")
        if len(row["pipeline_step_ms"]) != row["solver_steps"] or len(row["plan_step_ms"]) != row["solver_steps"]:
            raise ValueError("missing step timings")
        if row["nav_time"] > 25.+1e-6 or row["steps"][-1]["t"] > 25.+1e-6:
            raise ValueError("execution exceeded registered deadline")
        if metrics(row)[:3].sum() != 1:
            raise ValueError("terminal event does not partition SR/CR/TR")
        path = Path(row["provenance_file"])
        if path not in checked_manifests:
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["provenance_sha256"]:
                raise ValueError("provenance altered")
            manifest = json.loads(path.read_text())
            cfg = manifest["config"]
            for field, expected in (("horizon",t["candidate"]["horizon"]), ("dt",.25),
                                    ("v_max",1.), ("a_max",2.), ("omega_max",.8)):
                if cfg[field] != expected:
                    raise ValueError(f"runtime planner config differs: {field}")
            for file, sha in manifest["files"].items():
                if file in frozen["sources"] and sha != frozen["sources"][file]:
                    raise ValueError(f"episode used different source: {file}")
            checked_manifests.add(path)
        if t["candidate"]["family"] != "bayes":
            c = t["candidate"]
            setting = run.OUT / "settings" / f"{c['arm']}_{c['profile']}_risk{c['risk']:g}.yaml"
            if row["settings_sha256"] != frozen["sources"][str(setting.resolve())]:
                raise ValueError("episode loaded different native weights/risk")
        speeds = np.asarray([0.] + [s["speed"] for s in row["steps"]])
        if speeds.min() < -1e-6 or speeds.max() > 1.+1e-6 or np.abs(np.diff(speeds)).max() > .5+1e-6:
            raise ValueError("executed speed/acceleration bound violated")
        if any(abs(s["action_b"]) > .2+1e-6 for s in row["steps"]):
            raise ValueError("executed steering bound violated")


def analyze():
    frozen = json.loads((run.OUT / "frozen.json").read_text())
    protocol = json.loads((run.OUT / "protocol.json").read_text())
    if frozen["protocol_sha256"] != run.digest(protocol):
        raise ValueError("protocol changed since freeze")
    run.verify_sources(frozen["sources"])
    acceptance = json.loads((run.OUT / "acceptance.json").read_text())
    acceptance_review = run.OUT / "ACCEPTANCE_REVIEW.md"
    if not acceptance_review.is_file():
        raise ValueError("missing pre-test review of synthetic acceptance")
    test = json.loads((run.OUT / "test_results.json").read_text())
    stability = json.loads((run.OUT / "stability_results.json").read_text())
    latency = json.loads((run.OUT / "latency_results.json").read_text())
    for rows, n, split in ((test,3600,"T"),(stability,720,"S"),(latency,90,"L")):
        verify(rows, frozen, n, split)
    output = dict(selected=frozen["selected"], verification="complete, paired, frozen sources unchanged",
                  main={}, comparisons={}, stability={}, latency={},
                  synthetic_acceptance=[dict(family=r["candidate"]["family"], case=r["case"],
                    success=r["success"], collision=r["collision"], time=r["nav_time"]) for r in acceptance],
                  acceptance_review=str(acceptance_review))
    for occ, condition in ((True,"occluded"),(False,"fully_observed")):
        groups = {f: [r for r in test if r["task"]["occluded"] == occ and
                     r["task"]["candidate"]["family"] == f] for f in run.FAMILIES}
        if any(len(v) != 600 for v in groups.values()):
            raise ValueError("main table missing family/condition")
        output["main"][condition] = {f: summarize(rs) for f, rs in groups.items()}
        output["comparisons"][condition] = {f: paired(groups["bayes"], groups[f]) for f in ("tmpc","shmpc")}
        for f in ("tmpc","shmpc"):
            common = {key(r) for r in groups["bayes"] if r["result"]["success_without_overlap"]} & {
                key(r) for r in groups[f] if r["result"]["success_without_overlap"]}
            comparison = output["comparisons"][condition][f]
            comparison["joint_success_time_secondary"] = paired(
                [r for r in groups["bayes"] if key(r) in common],
                [r for r in groups[f] if key(r) in common]) if common else None
    primary = output["comparisons"]["occluded"]
    adjusted = holm([primary[f]["events"]["SR"]["p_block_sign_flip"] for f in ("tmpc","shmpc")])
    for f, p in zip(("tmpc","shmpc"), adjusted):
        primary[f]["events"]["SR"]["p_holm_two_primary"] = p
    repeated = stability + [r for r in test if r["task"]["occluded"] and r["task"]["case"] < 21020]
    for f in run.FAMILIES:
        output["stability"][f] = {str(seed): summarize([r for r in repeated if r["task"]["seed"] == seed
            and r["task"]["candidate"]["family"] == f]) for seed in (0,1,2)}
        output["latency"][f] = summarize([r for r in latency if r["task"]["candidate"]["family"] == f])
    output["stability_clustered_comparisons"] = {f: paired(
        [r for r in repeated if r["task"]["candidate"]["family"] == "bayes"],
        [r for r in repeated if r["task"]["candidate"]["family"] == f]) for f in ("tmpc","shmpc")}
    write_result(run.OUT / "analysis.json", output)
    lines = ["# Repaired Modern MPC Comparison", "", "Independent test only; 600 layouts per arm and observation condition.",
             "The controllers share legal-history predictions and a continuous accelerating unicycle executor.",
             "SR/CR/TR partition outcomes; collision takes precedence over timeout when both are recorded.",
             "Penalty time is arrival time on collision-free success and 25 s on failure.", ""]
    lines += ["The 600 layouts comprise 100 environment-seed blocks, each containing six scenes.",
              "Primary inference keeps each seed block intact; repeated planner seeds are not new independent environments.", ""]
    for condition, groups in output["main"].items():
        lines += [f"## {condition}", "", "| Controller | SR (%) | CR (%) | TR (%) | Penalty time (s) |", "|---|---:|---:|---:|---:|"]
        for f, stats in groups.items():
            c = next(c for c in frozen["selected"] if c["family"] == f)
            label = {"bayes":"Bayes-MPC", "tmpc":"T-MPC++", "shmpc":"SH-MPC"}[f]
            if f != "bayes":
                label += " (acados, goal-adapted)" if c["profile"] != "legacy" else " (repaired acados, legacy goal cost)"
            lines.append(f"| {label} | {100*stats['SR']:.2f} | {100*stats['CR']:.2f} | {100*stats['TR']:.2f} | {stats['penalty_time']:.3f} |")
        lines.append("")
    lines += ["## Limits", "", "- These are repaired acados implementations with disclosed goal-task profiles, not claimed bitwise reproductions of the authors' published experiments.",
        "- Synthetic acceptance was not an all-pass result: SH stopped before a stationary obstacle on a straight reference, while a known-static detour permitted arrival with the same settings. See ACCEPTANCE_REVIEW.md; the benchmark retains its fixed straight-reference protocol.",
        "- Risk parameters describe different events; identical numerical risk or equivalent safety certificates are not asserted.",
        "- Shared-posterior controller comparisons cannot by themselves establish that Bayesian estimation is indispensable.",
        "- Lower penalty time may result from fewer failures. Success-only times and paired arrival times on jointly successful cases are secondary, conditional analyses, not unconditional proof of faster travel.",
        "- Concurrent main-table timing is not the latency benchmark. Use the separate single-worker latency records.",
        "- Native solver success is not a safety certificate; report SH slack/support status and executed collision audit.",
        "- See analysis.json for scene-wise results, paired confidence intervals, Holm-adjusted primary tests and seed-clustered stability.", ""]
    (run.OUT / "MAIN_TABLE.md").write_text("\n".join(lines))
    print(json.dumps(output["main"], indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    analyze()
