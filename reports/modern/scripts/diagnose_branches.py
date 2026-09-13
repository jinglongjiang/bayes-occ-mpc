"""Trustworthy solver diagnostics, read from the fields that are actually written
(order section 13.4-C).

The main solver's `_info` cannot be used at a failing step.  When
`GuidanceConstraints::FindBestPlanner()` returns -1 the module returns early
(guidance_constraints.cpp:425) without executing
`_solver->_info = best_solver->_info`, so `qp_status`, `acados_status`,
`hpipm_raw_status`, `nlp_res` and `kkt_norm_inf` hold defaults or leftovers.
Reading them produced a plausible-looking status table that meant nothing; the
give-away was `nlp_res` sitting at ~7e-310, a denormal, and `kkt` at exactly 0.

The branch-level fields written at guidance_constraints.cpp:381-420 are set
*before* that early return, so they survive it.  This script reads those, from
the STEP reply the bridge already sends, and needs no change to any binary.

It answers three questions the failure analysis has to answer:

  * how many branches existed, were enabled, and succeeded, at a failing step;
  * what the non-guided branch (T-MPC++'s own plain LMPCC) did, since a failure
    of every guided branch while the plain one succeeds means something quite
    different from all of them failing;
  * what the branch-level equality residual actually was, which is the quantity
    the upstream failure rule thresholds at 1e-2.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, "/home/abc/temp/modern")

import numpy as np                                              # noqa: E402
from continuous_mpc_gate import DEFAULT_CROWDNAV, run_episode    # noqa: E402
from evaluate_matched_safety import config as point_config       # noqa: E402
from unicycle_mpc_gate import unicycle_config                    # noqa: E402
import modern_worker                                             # noqa: E402
from modern_worker import bridge_planner_factory                 # noqa: E402
from candidates import CANDIDATES, settings_for                  # noqa: E402

CASES = list(range(3660, 3672))          # debug layouts only


def collect(variant, candidate_id):
    candidate = next(c for c in CANDIDATES[variant]
                     if c.candidate_id == candidate_id)
    cfg = unicycle_config(point_config(2), horizon=candidate.horizon)
    factory = bridge_planner_factory(candidate.workspace,
                                     settings=str(settings_for(candidate)),
                                     path_overshoot=candidate.path_overshoot)
    records = []
    planner = None

    class Persist:
        def __call__(self, config):
            nonlocal planner
            if planner is None:
                planner = factory(config)
            else:
                planner.reset()
            return planner

    # The reply's keyed tail is parsed here rather than in the worker, so the
    # formal path and its code hash are untouched.
    original = modern_worker.BridgeController._exchange

    def capturing(self, line):
        reply = original(self, line)
        if line.startswith("STEP"):
            fields = {}
            for token in reply.split()[7:]:
                if "=" in token:
                    key, _, value = token.partition("=")
                    fields[key] = value
            fields["success"] = reply.split()[1]
            records.append(fields)
        return reply

    modern_worker.BridgeController._exchange = capturing
    try:
        for case in CASES:
            run_episode(DEFAULT_CROWDNAV, "bayes", 5, "circle_crossing", case, cfg,
                        4.0, None, 0, 0.0, 0.0, 1.0, time_limit=25,
                        planner_type=Persist(), robot_kinematics="unicycle")
    finally:
        modern_worker.BridgeController._exchange = original
        if planner is not None:
            planner._controller.close()
    return records


def summarise(name, records):
    failing = [r for r in records if r.get("success") == "0"]
    print(f"\n{name}: {len(records)} 步，其中失败 {len(failing)} 步 "
          f"({100*len(failing)/max(len(records),1):.1f}%)")
    if not failing:
        return {}

    def number(row, key, default=float("nan")):
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return default

    branches = collections.Counter(
        (row.get("br_total"), row.get("br_enabled"), row.get("br_ok"))
        for row in failing)
    print("  分支 (总数, 启用, 成功) 分布：")
    for key, count in branches.most_common(6):
        print(f"    {str(key):24s} {count:5d}  {100*count/len(failing):5.1f}%")

    original_branch = collections.Counter(
        (row.get("orig_disabled"), row.get("orig_exit"), row.get("orig_status"))
        for row in failing)
    print("  无引导分支 (禁用, 退出码, acados状态) 分布：")
    for key, count in original_branch.most_common(6):
        print(f"    {str(key):24s} {count:5d}  {100*count/len(failing):5.1f}%")

    residuals = np.array([number(row, "br_res_eq") for row in failing])
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size:
        print(f"  分支级 res_eq（上游阈值 1e-2），{residuals.size} 个有效值：")
        for q in (50, 90, 99, 100):
            print(f"    p{q:<3d} {np.percentile(residuals, q):.4e}")
        over = 100.0 * float((residuals > 1e-2).mean())
        print(f"    超过 1e-2 的比例 {over:.1f}%")

    statuses = collections.Counter(row.get("br_status") for row in failing)
    print(f"  分支 acados 状态：{dict(statuses.most_common(6))}")
    nonfinite = collections.Counter(
        (row.get("nf_par"), row.get("nf_ws")) for row in failing)
    print(f"  非有限 (参数, 热启动) 计数：{dict(nonfinite.most_common(4))}")
    return {"failing": len(failing), "total": len(records),
            "branches": {str(k): v for k, v in branches.items()},
            "original_branch": {str(k): v for k, v in original_branch.items()},
            "br_res_eq_p50": float(np.percentile(residuals, 50)) if residuals.size else None,
            "br_res_eq_over_1e-2_pct": float((residuals > 1e-2).mean() * 100)
                                       if residuals.size else None,
            "br_status": {str(k): v for k, v in statuses.items()}}


def main():
    out = {}
    for variant in ("tmpc", "shmpc"):
        records = collect(variant, "c01_author")
        out[variant] = summarise(f"{variant} c01_author", records)
    Path("/home/abc/temp/modern/snapshot/branch_diagnostics.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print("\n-> snapshot/branch_diagnostics.json")


if __name__ == "__main__":
    main()
