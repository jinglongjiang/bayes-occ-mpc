"""Freeze the configuration before any T evaluation (order section 13.5).

Nothing in the formal matrix may be chosen after a T outcome has been seen, so
this writes down, in one file and before T starts, every input that decides what
a formal episode computes: which candidate each arm runs, the model identity and
time assumptions, the actuator box, the thread budget, the RNG policy, the case
lists, the statistics method, and the SHA256 of every build artefact and source
file involved.

The rule about the working point is also enforced here rather than trusted: an
arm may only be frozen at the candidate D2 selected.  There is no "D2 did not
pick it but the report uses it anyway".

`bayes_r1` is not selected separately.  It inherits the Bayesian family's frozen
candidate and differs only by `existence_override = 1.0`, which is read where
the risk term weights each track and nowhere else.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/abc/temp/modern")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")

from candidates import CANDIDATES, settings_for                       # noqa: E402

ROOT = Path("/home/abc/temp/modern")
BUILD = Path("/home/abc/workspace/bayes_occ_mpc/build_modern")
SRC = Path("/home/abc/workspace/bayes_occ_mpc")

SOURCES = [
    SRC / "continuous_mpc_gate.py", SRC / "unicycle_mpc_gate.py",
    SRC / "modern_worker.py", SRC / "evaluate_matched_safety.py",
    ROOT / "candidates.py", ROOT / "case_registry.py", ROOT / "analysis.py",
    ROOT / "freeze.py", ROOT / "env.sh", ROOT / "build_variant.sh",
    ROOT / "deploy_bridge.sh", ROOT / "bridge/mpc_bridge.cpp",
    ROOT / "bridge/run_bridge_cohort.py", ROOT / "bridge/verify_uncertainty.py",
    ROOT / "risk_sweep/front.py",
    SRC / "bayesian_rfs.py", ROOT / "pipeline.sh", ROOT / "supervise.py",
    ROOT / "snapshot/candidates.json", ROOT / "snapshot/case_registry.json",
    ROOT / "snapshot/d1_selection.json", ROOT / "snapshot/d2_selection.json",
    ROOT / "snapshot/acceptance_takeover.json",
    ROOT / "bridge/interface_audit.py", ROOT / "test_takeover.py",
    ROOT / "snapshot/dynamics_crosscheck.json", ROOT / "dynamics_crosscheck.py",
    BUILD / "shmpc/src/scenario_module/config/params.yaml",
]


def sha256(path: Path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_artefacts(name):
    mp = BUILD / name / "src/mpc_planner"
    files = {
        "settings.yaml": mp / "mpc_planner_jackalsimulator/config/settings.yaml",
        "generator.py": mp / "mpc_planner_jackalsimulator/scripts/generate_jackalsimulator_solver.py",
        "acados_ocp_solver.so": mp / "mpc_planner_solver/acados/Solver/libacados_ocp_solver_Solver.so",
        "libmpc_planner_solver.so": BUILD / name / "devel/lib/libmpc_planner_solver.so",
        "libmpc_planner_modules.so": BUILD / name / "devel/lib/libmpc_planner_modules.so",
        "mpc_bridge": BUILD / name / "devel/lib/mpc_planner_bridge/mpc_bridge",
        "bridge_source": BUILD / name / "src/mpc_planner_bridge/src/mpc_bridge.cpp",
    }
    for source in (mp / "mpc_planner_jackalsimulator/config").glob("*.yaml"):
        files[f"config/{source.name}"] = source
    # Capture the libraries actually loaded, not only the main executable.
    executable = files["mpc_bridge"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(BUILD / name / "devel/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(["ldd", str(executable)], env=env, text=True,
                            capture_output=True, check=True)
    if "not found" in result.stdout:
        raise RuntimeError(f"unresolved library for {name}: {result.stdout}")
    for line in result.stdout.splitlines():
        words = line.split()
        if "=>" in words and len(words) > 2 and words[2].startswith("/"):
            files[f"loaded/{words[0]}"] = Path(words[2])
    return {label: {"path": str(path), "sha256": sha256(path)}
            for label, path in files.items()}


def verify_frozen():
    target = ROOT / "snapshot/frozen.json"
    frozen = json.loads(target.read_text())
    files = dict(frozen["sources"])
    for spec in frozen["arms_occluded"].values():
        if "settings" in spec:
            files[spec["settings"]] = spec["settings_sha256"]
        for entry in spec.get("artefacts", {}).values():
            files[entry["path"]] = entry["sha256"]
    bad = [name for name, expected in files.items()
           if expected is None or sha256(Path(name)) != expected]
    if bad:
        raise RuntimeError(f"frozen inputs changed or missing: {bad[:8]}")
    return frozen


def main():
    target = ROOT / "snapshot/frozen.json"
    if target.exists():
        verify_frozen()
        print("Existing freeze verified; not overwriting timestamp or configuration")
        return
    for queue in ("T-OCC", "T-FULL"):
        if list((ROOT / "runs" / queue).glob("*.jsonl*")):
            raise RuntimeError("test output exists before freezing")
    sys.path.insert(0, str(ROOT / "bridge"))
    from run_bridge_cohort import validated_rows
    for queue in ("D1", "D2"):
        rows = validated_rows(queue)
        if any(r["event"] == "error" for r in rows):
            raise RuntimeError(f"unresolved development errors in {queue}")
    acceptance = json.loads((ROOT / "snapshot/acceptance_takeover.json").read_text())
    if not acceptance.get("proceed_shared_belief_comparison"):
        raise RuntimeError("acceptance has not authorized this comparison")
    selection = json.loads((ROOT / "snapshot/d2_selection.json").read_text())
    chosen = selection["selected"]

    arms_occluded, arms_full = {}, {}
    for arm, family, override in (("bayes_full", "bayes", None),
                                  ("bayes_r1", "bayes", 1.0),
                                  ("tmpc_shared", "tmpc", None),
                                  ("shmpc_shared", "shmpc", None)):
        candidate_id = chosen[family]
        candidate = next(c for c in CANDIDATES[family] if c.candidate_id == candidate_id)
        spec = {"family": family, "candidate_id": candidate_id,
                "existence_override": override,
                "horizon": candidate.horizon,
                "prediction_seconds": candidate.horizon * 0.25,
                "rationale": candidate.rationale}
        if family != "bayes":
            spec.update({
                "workspace": candidate.workspace,
                "risk_own_units": candidate.risk,
                "risk_effective": candidate.effective_risk,
                "contour_weight": candidate.contour,
                "terminal_contouring_weight": candidate.terminal_contouring,
                "path_overshoot_m": candidate.path_overshoot,
                "settings": str(settings_for(candidate)),
                "settings_sha256": sha256(settings_for(candidate)),
                "artefacts": workspace_artefacts(candidate.workspace)})
        else:
            spec.update({"risk_point": candidate.risk_point,
                         "population": candidate.population,
                         "iterations": candidate.iterations})
        arms_occluded[arm] = spec
        if arm != "bayes_r1":
            arms_full[arm] = dict(spec)

    frozen = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "selection_source": "snapshot/d2_selection.json",
        "selection_rule": selection["rule"],
        "arms_occluded": arms_occluded,
        "arms_full": arms_full,
        "contract": {
            "robot_visible": False, "robot_radius": 0.3, "human_radius": 0.3,
            "human_margin": 0.10, "v_max": 1.0, "a_max": 2.0, "omega_max": 0.8,
            "reverse": False, "execution_period_s": 0.25, "time_limit_s": 25,
            "goal_radius_m": 0.25, "auxiliary_goal_radii_m": [0.5, 1.0],
            "actuator_box_enforcement":
                "saturated at the environment boundary for every arm alike, "
                "with violations counted per episode; tolerance 1e-6",
        },
        "resources": {
            "machine": "local Ryzen 7 5700G, 8 physical / 16 logical cores, 15 GB",
            "workers": int(os.environ.get("WORKERS", "1")), "threads_per_worker": 4,
            "development_legacy_workers": 2,
            "development_note": "CC D1 is retained as coarse development only; its environment specified OMP_NUM_THREADS but did not enforce OMP_THREAD_LIMIT. D2 and all formal queues enforce four threads; L includes a separately enforced one-thread profile. D1 timings are not hardware benchmarks.",
            "note": "the 4090 carries the Bayesian project only and has neither "
                    "ROS nor acados nor the free disk to install them, so the "
                    "whole matrix runs on one machine; splitting one method per "
                    "machine is ruled out by the order in any case",
        },
        "rng_policy": {
            "environment": "seeded by case_id",
            "planner": "case_id * 100003 + step * 97 + 1729 + repeat * 10000019",
            "comparators":
                "SH-MPC seeds its scenario sampler from std::random_device and "
                "T-MPC++ varies through its parallel branches, so neither is "
                "bit-reproducible.  Native parallelism is kept under a common "
                "CPU quota, as the order prefers, and the variation is bounded "
                "by three independent repeats per layout rather than hidden "
                "behind a single-thread run.",
        },
        "statistics": {
            "primary": "macro over six scenes of SR, CR, TR and penalised time",
            "aggregation": "repeats averaged within a layout first",
            "resampling": "cluster bootstrap over layouts within scene, "
                          "10000 draws, seed 20260909",
            "p_values": "two-sided paired layout-cluster sign permutation with plus-one correction",
            "correction": "Holm-Bonferroni over the six pre-registered tests",
        },
        "cases": {name: {"n": entry["n"], "cases": entry["cases"]}
                  for name, entry in json.loads(
                      (ROOT / "snapshot/case_registry.json").read_text()
                  )["splits"].items()},
        "acceptance_scope": acceptance,
        "legacy_development_sources": str(ROOT / "takeover_20260909/original"),
        "sources": {str(path): sha256(path) for path in SOURCES},
    }

    from continuous_mpc_gate import DEFAULT_CROWDNAV
    for folder in (DEFAULT_CROWDNAV / "crowd_sim", DEFAULT_CROWDNAV / "crowd_nav"):
        for pattern in ("*.py", "*.config"):
            for path in folder.rglob(pattern):
                if not any(p in path.parts for p in ("runs", "__pycache__", "backup")):
                    frozen["sources"][str(path)] = sha256(path)
    archive = ROOT / "snapshot/frozen_files"
    all_files = dict(frozen["sources"])
    for spec in arms_occluded.values():
        if "settings" in spec:
            all_files[spec["settings"]] = spec["settings_sha256"]
        for item in spec.get("artefacts", {}).values():
            all_files[item["path"]] = item["sha256"]
    for name, expected in all_files.items():
        if expected is None:
            raise RuntimeError(f"missing freeze input: {name}")
        dest = archive / Path(name).relative_to("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
        if sha256(dest) != expected:
            raise RuntimeError(f"source changed during snapshot: {name}")
    frozen["archive"] = str(archive)
    target.write_text(json.dumps(frozen, indent=1, ensure_ascii=False))
    verify_frozen()
    print("已冻结工作点：")
    for arm, spec in arms_occluded.items():
        extra = (f"risk={spec['risk_own_units']}(有效{spec['risk_effective']}) "
                 f"contour={spec['contour_weight']} overshoot={spec['path_overshoot_m']}"
                 if spec["family"] != "bayes"
                 else f"point={spec['risk_point']} pop={spec['population']} "
                      f"iters={spec['iterations']} override={spec['existence_override']}")
        print(f"  {arm:14s} {spec['family']:6s} {spec['candidate_id']:18s} "
              f"N={spec['horizon']:2d} ({spec['prediction_seconds']:.0f}s)  {extra}")
    print(f"\n-> {target}")
    print(f"登记源文件 {len(SOURCES)} 个，构建产物按工作副本逐项哈希")


if __name__ == "__main__":
    main()
