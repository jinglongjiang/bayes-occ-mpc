"""Statistics and paper tables for the formal queues (order section 13.9).

Three things this file is careful about, because each is a way to make a
comparison look stronger than it is:

  * The three planning repeats share a layout.  Treating 1800 episodes as 1800
    independent trials would be wrong, so a layout is reduced to a mean over its
    repeats first, and the bootstrap resamples layouts -- clusters -- inside
    each scene.  The per-repeat McNemar counts are reported too, as an auxiliary
    view, but no seed is chosen from them.

  * The main comparison is pre-registered: occluded `bayes_full` against each
    comparator, on collision rate, success rate and penalised time.  Those six
    tests carry a Holm correction.  Everything else -- the existence-probability
    control, the full-observation condition, per-scene breakdowns -- is
    explicitly auxiliary and is not promoted into the main claim.

  * "Not significant" is not "equivalent".  No non-inferiority claim is made
    here, because no margin was registered; where a difference is uncertain the
    exported text says the safety difference is undetermined rather than equal.

The primary score of an arm is the macro average over the six scenes, weighted
equally, of: success without overlap, the union collision indicator, timeout,
and penalised time (nav_time when the episode succeeded safely, else 25 s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/abc/temp/modern")
PENALTY = 25.0
BOOTSTRAP = 10000
SEED = 20260909
SCENES = ("baseline_circle", "baseline_square", "dense_circle",
          "dense_square", "large_circle", "large_square")

METRICS = ("sr", "cr", "tr", "time")
METRIC_LABEL = {"sr": "SR", "cr": "CR", "tr": "TR", "time": "罚时"}


# ------------------------------------------------------------------ loading

def load(queue):
    path = ROOT / "snapshot" / f"{queue}_episodes.jsonl"
    if not path.exists():
        raise SystemExit(f"缺少 {path}；先跑 run_bridge_cohort.py aggregate {queue}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episode_values(row):
    """The four primary quantities for one episode, as an audited, mutually
    exclusive outcome: a collision found by the audit wins over the environment's
    own label, so the four rates sum to 100%."""
    collision = bool(row.get("collision_union"))
    error = row["event"] == "error" and not collision
    success = bool(row.get("success_without_overlap")) and not error and not collision
    timeout = (not success) and (not collision) and (not error)
    return {
        "sr": float(success),
        "cr": float(bool(collision)),
        "tr": float(timeout),
        "err": float(error),
        "time": float(row["nav_time"]) if success else PENALTY,
        "cr_unknown": float(error and row.get("collision_union") is None),
    }


def by_layout(rows, arm, condition):
    """{scene: {case: {metric: mean over repeats}}} for one arm."""
    out = {}
    for row in rows:
        if row["arm"] != arm or row["condition"] != condition:
            continue
        scene = out.setdefault(row["scene_id"], {})
        scene.setdefault(row["case_id"], []).append(episode_values(row))
    return {scene: {case: {m: float(np.mean([v[m] for v in values]))
                           for m in ("sr", "cr", "tr", "err", "time", "cr_unknown")}
                    for case, values in cases.items()}
            for scene, cases in out.items()}


# ---------------------------------------------------------------- estimates

def macro(layouts, metric, keys=None):
    """Equally weighted mean over scenes of the layout-level means."""
    per_scene = []
    for scene in sorted(layouts):
        cases = keys[scene] if keys else sorted(layouts[scene])
        values = [layouts[scene][c][metric] for c in cases if c in layouts[scene]]
        if values:
            per_scene.append(float(np.mean(values)))
    return float(np.mean(per_scene)) if per_scene else float("nan")


def paired_bootstrap(a_layouts, b_layouts, metric, rng):
    """Cluster bootstrap over layouts, paired: the same resampled layouts are
    used for both arms, so the difference keeps the pairing the design has."""
    if set(a_layouts) != set(b_layouts) or any(
            set(a_layouts[s]) != set(b_layouts[s]) for s in a_layouts):
        raise ValueError("paired layouts differ; missing pairs must not be silently dropped")
    scenes = sorted(a_layouts)
    common = {s: sorted(a_layouts[s]) for s in scenes}
    point = macro(a_layouts, metric, common) - macro(b_layouts, metric, common)
    draws = np.zeros(BOOTSTRAP)
    permuted = np.zeros(BOOTSTRAP)
    for s in scenes:
        delta = np.asarray([a_layouts[s][c][metric] - b_layouts[s][c][metric]
                            for c in common[s]])
        draws += delta[rng.integers(len(delta), size=(BOOTSTRAP, len(delta)))].mean(axis=1) / len(scenes)
        # Swap whole layout clusters, including all repeated planner runs.
        signs = 2 * rng.integers(2, size=(BOOTSTRAP, len(delta))) - 1
        permuted += (signs * delta).mean(axis=1) / len(scenes)
    low, high = np.percentile(draws, [2.5, 97.5])
    p = float((1 + np.count_nonzero(np.abs(permuted) >= abs(point) - 1e-12)) / (BOOTSTRAP + 1))
    return {"difference": point, "ci_low": float(low), "ci_high": float(high),
            "p": min(1.0, p), "n_layouts": int(sum(len(v) for v in common.values()))}


def holm(tests):
    """Holm-Bonferroni over the pre-registered family."""
    ordered = sorted(tests, key=lambda t: tests[t]["p"])
    m = len(ordered)
    previous = 0.0
    out = {}
    for rank, name in enumerate(ordered):
        adjusted = max(previous, min(1.0, (m - rank) * tests[name]["p"]))
        previous = adjusted
        out[name] = adjusted
    return out


def mcnemar_counts(rows, arm_a, arm_b, condition, metric="sr"):
    """Auxiliary per-repeat discordant counts.  Reported, not used to choose."""
    def index(arm):
        return {(r["scene_id"], r["case_id"], r["repeat"]): episode_values(r)[metric]
                for r in rows if r["arm"] == arm and r["condition"] == condition}
    a, b = index(arm_a), index(arm_b)
    shared = set(a) & set(b)
    per_repeat = {}
    for key in shared:
        cell = per_repeat.setdefault(key[2], {"a_only": 0, "b_only": 0, "both": 0, "neither": 0})
        if a[key] > b[key]:
            cell["a_only"] += 1
        elif b[key] > a[key]:
            cell["b_only"] += 1
        elif a[key] > 0:
            cell["both"] += 1
        else:
            cell["neither"] += 1
    return per_repeat


# ------------------------------------------------------------------ exports

def summary_table(rows, arms, condition):
    table = {}
    for arm in arms:
        layouts = by_layout(rows, arm, condition)
        if not layouts:
            continue
        entry = {"per_scene": {}, "macro": {}}
        for scene in sorted(layouts):
            cases = sorted(layouts[scene])
            entry["per_scene"][scene] = {
                m: float(np.mean([layouts[scene][c][m] for c in cases]))
                for m in ("sr", "cr", "tr", "err", "time", "cr_unknown")}
            entry["per_scene"][scene]["layouts"] = len(cases)
            entry["per_scene"][scene]["episodes"] = sum(
                1 for r in rows if r["arm"] == arm and r["condition"] == condition
                and r["scene_id"] == scene)
        for m in ("sr", "cr", "tr", "err", "time", "cr_unknown"):
            entry["macro"][m] = macro(layouts, m)
        rng = np.random.default_rng(SEED)
        draws = np.zeros((BOOTSTRAP, 3))
        for scene, cases in layouts.items():
            matrix = np.asarray([[cases[c][m] for m in ("sr", "cr", "time")] for c in sorted(cases)])
            draws += matrix[rng.integers(len(matrix), size=(BOOTSTRAP, len(matrix)))].mean(axis=1) / len(layouts)
        entry["macro_ci"] = {m: list(map(float, np.percentile(draws[:, i], [2.5, 97.5])))
                             for i, m in enumerate(("sr", "cr", "time"))}
        table[arm] = entry
    return table


def write_csv(table, path):
    lines = ["arm,scene,layouts,episodes,SR,CR,TR,ERR,CR_unknown,penalised_time,SR_ci_low,SR_ci_high,CR_ci_low,CR_ci_high,time_ci_low,time_ci_high"]
    for arm, entry in table.items():
        for scene, values in entry["per_scene"].items():
            lines.append(f"{arm},{scene},{values['layouts']},{values['episodes']},"
                         f"{100*values['sr']:.2f},{100*values['cr']:.2f},"
                         f"{100*values['tr']:.2f},{100*values['err']:.2f},"
                         f"{100*values['cr_unknown']:.2f},{values['time']:.3f},,,,,,")
        m = entry["macro"]
        lines.append(f"{arm},MACRO,,,{100*m['sr']:.2f},{100*m['cr']:.2f},"
                     f"{100*m['tr']:.2f},{100*m['err']:.2f},"
                     f"{100*m['cr_unknown']:.2f},{m['time']:.3f},"
                     + ",".join(f"{value * (1 if metric == 'time' else 100):.3f}"
                                for metric in ("sr", "cr", "time") for value in entry["macro_ci"][metric]))
    path.write_text("\n".join(lines) + "\n")


def write_tex(table, path, caption, label):
    head = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}",
            f"\\label{{{label}}}", "\\small",
            "\\begin{tabular}{lrrrrr}", "\\toprule",
            "Method & SR (\\%) & CR (\\%) & TR (\\%) & Err (\\%) & "
            "Penalised time (s) \\\\", "\\midrule"]
    body = []
    for arm, entry in table.items():
        m = entry["macro"]
        body.append(f"{arm.replace('_', ' ')} & {100*m['sr']:.1f} & {100*m['cr']:.1f} "
                    f"& {100*m['tr']:.1f} & {100*m['err']:.1f} & {m['time']:.2f} \\\\")
    tail = ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(head + body + tail) + "\n")


def goal_sensitivity(rows, path):
    lines = ["arm,condition,radius,first_entry_rate,clean_first_entry_rate,median_time_s"]
    for arm in sorted({r["arm"] for r in rows}):
        for condition in sorted({r["condition"] for r in rows}):
            group = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
            if not group:
                continue
            for radius in ("0.25", "0.5", "1.0"):
                entered = [r for r in group if (r.get("goal_entry") or {}).get(radius) is not None]
                clean = [r for r in entered if (r.get("goal_entry_clean") or {}).get(radius)]
                times = [r["goal_entry"][radius] for r in entered]
                lines.append(f"{arm},{condition},{radius},"
                             f"{100*len(entered)/len(group):.2f},"
                             f"{100*len(clean)/len(group):.2f},"
                             f"{np.median(times) if times else float('nan'):.2f}")
    path.write_text("\n".join(lines) + "\n")


def failure_summary(rows, path):
    lines = ["arm,condition,episodes,solver_failure_steps,steps,"
             "solved_but_no_progress,stalled_near_goal,collisions,timeouts,errors,"
             "bound_violation_steps,max_bound_excess"]
    for arm in sorted({r["arm"] for r in rows}):
        for condition in sorted({r["condition"] for r in rows}):
            group = [r for r in rows if r["arm"] == arm and r["condition"] == condition]
            if not group:
                continue
            failures = steps = stalled = no_progress = 0
            for row in group:
                trace = row.get("steps") or []
                steps += len(trace)
                failures += sum(1 for s in trace if s.get("solver_success") == 0.0)
                if trace and row["event"] == "timeout":
                    tail = trace[-20:]
                    if all(s["advance"] < 0.01 for s in tail):
                        no_progress += 1
                        if tail[-1]["goal_distance"] < 1.5:
                            stalled += 1
            lines.append(
                f"{arm},{condition},{len(group)},{failures},{steps},{no_progress},"
                f"{stalled},{sum(1 for r in group if r.get('collision_union'))},"
                f"{sum(1 for r in group if r['event']=='timeout')},"
                f"{sum(1 for r in group if r['event']=='error')},"
                f"{sum(r.get('bound_violations',0) for r in group)},"
                f"{max([r.get('max_bound_excess',0.0) for r in group]):.4f}")
    path.write_text("\n".join(lines) + "\n")


def latency(rows, path):
    lines = ["arm,condition,threads,steps,plan_p50,plan_p95,plan_p99,"
             "pipeline_p50,pipeline_p95,pipeline_p99,over_250ms_rate"]
    for arm, condition, threads in sorted({(r["arm"], r["condition"], r["threads"]) for r in rows}):
            group = [r for r in rows if (r["arm"], r["condition"], r["threads"]) == (arm, condition, threads)]
            plans = [np.asarray(r["plan_step_ms"]) for r in group if r.get("plan_step_ms")]
            pipes = [np.asarray(r["pipeline_step_ms"]) for r in group if r.get("pipeline_step_ms")]
            plan = np.concatenate(plans) if plans else np.array([])
            pipe = np.concatenate(pipes) if pipes else np.array([])
            if plan.size == 0:
                continue
            # The deadline denominator is the 250 ms execution period, never the
            # method's own prediction step.
            lines.append(
                f"{arm},{condition},{threads},{plan.size},"
                + ",".join(f"{np.percentile(plan, q):.2f}" for q in (50, 95, 99)) + ","
                + ",".join(f"{np.percentile(pipe, q):.2f}" for q in (50, 95, 99)) + ","
                + f"{100.0*float((pipe > 250.0).mean()):.3f}")
    path.write_text("\n".join(lines) + "\n")


def successful_efficiency(rows, path):
    lines = ["condition,comparison,paired_success_episodes,reference_time,comparator_time,reference_path,comparator_path"]
    for condition in sorted({r["condition"] for r in rows}):
        reference = {(r["scene_id"], r["case_id"], r["repeat"]): r for r in rows
                     if r["condition"] == condition and r["arm"] == "bayes_full"
                     and episode_values(r)["sr"]}
        for arm in sorted({r["arm"] for r in rows} - {"bayes_full"}):
            other = {(r["scene_id"], r["case_id"], r["repeat"]): r for r in rows
                     if r["condition"] == condition and r["arm"] == arm
                     and episode_values(r)["sr"]}
            common = sorted(reference.keys() & other.keys())
            if not common:
                continue
            values = [np.mean([index[k][field] for k in common])
                      for field in ("nav_time", "path_length") for index in (reference, other)]
            lines.append(f"{condition},bayes_full_vs_{arm},{len(common)}," + ",".join(f"{v:.4f}" for v in values))
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- main

def main():
    sys.path.insert(0, str(ROOT / "bridge"))
    from run_bridge_cohort import validated_rows
    from freeze import verify_frozen
    verify_frozen()
    out = ROOT / "final"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    occ = validated_rows("T-OCC")
    full = validated_rows("T-FULL")
    latency_rows = validated_rows("L")
    arms_occ = sorted({r["arm"] for r in occ})
    arms_full = sorted({r["arm"] for r in full})

    occ_table = summary_table(occ, arms_occ, "occluded")
    full_table = summary_table(full, arms_full, "full")
    write_csv(occ_table, out / "main_occlusion.csv")
    write_tex(occ_table, out / "main_occlusion.tex",
              "Occluded six-scene comparison, macro averaged over scenes. "
              "All arms share one detector, tracker and marginal predictions; "
              "the comparators are shared-belief adaptations of the published "
              "planners, not the authors' original systems.",
              "tab:main_occlusion")
    write_csv(full_table, out / "full_observation.csv")
    write_tex(full_table, out / "full_observation.tex",
              "Full-observation control, same frozen working points.",
              "tab:full_observation")

    reference = "bayes_full"
    comparators = [a for a in arms_occ if a in ("tmpc_shared", "shmpc_shared")]
    primary = {}
    for comparator in comparators:
        a = by_layout(occ, reference, "occluded")
        b = by_layout(occ, comparator, "occluded")
        for metric in ("cr", "sr", "time"):
            primary[f"{reference}_vs_{comparator}|{metric}"] = paired_bootstrap(
                a, b, metric, rng)
    adjusted = holm(primary) if primary else {}
    for name, value in adjusted.items():
        primary[name]["p_holm"] = value

    auxiliary = {}
    for comparator in comparators:
        auxiliary[f"mcnemar|{reference}_vs_{comparator}"] = mcnemar_counts(
            occ, reference, comparator, "occluded")
    for arm in arms_occ:
        if arm not in (reference, *comparators):
            a = by_layout(occ, reference, "occluded")
            b = by_layout(occ, arm, "occluded")
            for metric in ("cr", "sr", "time"):
                auxiliary[f"{reference}_vs_{arm}|{metric}"] = paired_bootstrap(
                    a, b, metric, rng)

    (out / "paired_comparisons.json").write_text(json.dumps(
        {"preregistered_family": list(primary),
         "correction": "Holm-Bonferroni over the six pre-registered tests",
         "bootstrap": {"iterations": BOOTSTRAP, "seed": SEED,
                       "cluster": "layout within scene, repeats aggregated first"},
         "p_values": "two-sided paired layout-cluster sign permutation, plus-one Monte Carlo correction",
         "primary": primary, "auxiliary": auxiliary}, indent=1, ensure_ascii=False))

    everything = occ + full
    goal_sensitivity(everything, out / "goal_sensitivity.csv")
    failure_summary(everything, out / "failure_summary.csv")
    successful_efficiency(everything, out / "common_success_efficiency_AUXILIARY.csv")
    latency(latency_rows, out / "latency.csv")
    latency(everything, out / "latency_evaluation_load.csv")
    (out / "main_summaries.json").write_text(json.dumps(
        {"occluded": occ_table, "full": full_table}, indent=1))

    print("主表（遮挡，六场景等权 macro）：")
    print(f"  {'arm':16s} {'SR%':>6s} {'CR%':>6s} {'TR%':>6s} {'ERR%':>6s} {'罚时s':>7s}")
    for arm, entry in occ_table.items():
        m = entry["macro"]
        print(f"  {arm:16s} {100*m['sr']:6.1f} {100*m['cr']:6.1f} {100*m['tr']:6.1f} "
              f"{100*m['err']:6.1f} {m['time']:7.2f}")
    print("\n预注册主对比（Holm 校正后）：")
    for name, value in primary.items():
        print(f"  {name:42s} 差值 {value['difference']:+8.4f}  "
              f"95%CI [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]  "
              f"p={value['p']:.4f}  p_holm={value.get('p_holm', float('nan')):.4f}")
    print(f"\n导出 -> {out}")


if __name__ == "__main__":
    main()
