"""Build and verify the D1 / D2 / T case registry (order section 13.5).

A case id is not evidence of a distinct layout: the generator seeds from it, and
if the seeding wraps or ignores the id the "independent layouts" are the same
scene repeated.  So this hashes the layout the environment actually builds --
robot start and goal, and every pedestrian's start, goal, preferred speed and
radius -- and refuses to register a split that contains duplicates or that
overlaps another split.

Splits, non-overlapping by construction and checked by hash:

  debug   3650-3749  everything already looked at.  Excluded from D1, D2 and T.
  D1      11000-11039  40 five-person layouts, half circle and half square.
  D2      11100-11199  100 further five-person layouts, disjoint from D1.
  T       12000-12099  100 layouts, instantiated in each of the six registered
                     scenes; the scene configuration is part of the pairing key,
                     so the same id in two scenes is two different layouts.

Development is five-person only, per the order; the 10/12/20-person scenes exist
only in T and are never seen while a working point is chosen.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")

import numpy as np                                              # noqa: E402
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env      # noqa: E402
from evaluate_matched_safety import SCENES                       # noqa: E402

OUT = Path("/home/abc/temp/modern/snapshot")

# Every id any earlier run of this project consumed, rebuilt by scanning the
# result files rather than trusted to memory.  The first version of this file
# put D1 at 4000-4039, D2 at 4100-4199 and T at 5000-5099; the scan showed all
# three had been used before -- 5000-5099 is exactly the six-scene set in
# temp/formal/D, so calling it an independent test set would have been false.
# The splits therefore live above every historical id, and `main` refuses to
# register a split that overlaps history, so the mistake cannot repeat quietly.
HISTORY_ROOTS = ("/home/abc/temp", "/home/abc/workspace/bayes_occ_mpc/results",
                 "/home/abc/workspace/bayes_occ_mpc")

DEBUG = list(range(3650, 3750))          # already-seen debug layouts
D1_CIRCLE = list(range(11000, 11020))
D1_SQUARE = list(range(11020, 11040))
D2_CIRCLE = list(range(11100, 11150))
D2_SQUARE = list(range(11150, 11200))
# The exclusive latency window needs five layouts in each of the six scenes.
# They are development-side, never test, and never enter T.
L_CASES = list(range(11200, 11205))
T_CASES = list(range(12000, 12100))

# Development scenes: five people, circle and square in equal share.  The widths
# match the registered six-scene contract for their generator.
DEV_CIRCLE = ("dev_circle", 5, "circle_crossing", 4.0, None)
DEV_SQUARE = ("dev_square", 5, "square_crossing", None, 10.0)


def historical_case_ids():
    """Case ids consumed by any earlier experiment in this project.

    Reads `case_id` fields and integer `cases` lists out of every JSON/JSONL
    result file, skipping this round's own run directory.  A large case integer
    is not evidence of new data, and neither is a range nobody remembers using.
    """
    used = set()

    def harvest(obj):
        if isinstance(obj, dict):
            value = obj.get("case_id")
            if isinstance(value, int):
                used.add(value)
            cases = obj.get("cases")
            if isinstance(cases, list) and cases and all(isinstance(c, int) for c in cases):
                used.update(cases)
            for item in obj.values():
                harvest(item)
        elif isinstance(obj, list):
            for item in obj:
                harvest(item)

    paths = set()
    for root in HISTORY_ROOTS:
        for pattern in ("**/*.json", "**/*.jsonl"):
            paths.update(Path(root).glob(pattern))
    for path in paths:
        text = str(path)
        if text.startswith("/home/abc/temp/modern/"):
            continue          # includes this round's manifests and snapshots
        try:
            if path.stat().st_size > 300_000_000:
                continue
            if path.suffix == ".jsonl":
                with open(path) as handle:
                    for line in handle:
                        if line.strip():
                            harvest(json.loads(line))
            else:
                harvest(json.loads(path.read_text()))
        except (ValueError, OSError):
            continue
    return used


def scene_hash(scene) -> str:
    """The full scene configuration, not just the crowd size.  Two square scenes
    that differ only in width previously collapsed onto one another in the
    pairing key; including every field is what stops that."""
    scene_id, humans, generator, radius, width = scene
    payload = json.dumps({"scene_id": scene_id, "human_num": humans,
                          "scenario": generator, "circle_radius": radius,
                          "square_width": width}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def layout_hash(env) -> str:
    """A hash of the initial state the environment actually built."""
    rows = [[env.robot.px, env.robot.py, env.robot.gx, env.robot.gy,
             env.robot.v_pref, env.robot.radius]]
    for human in env.humans:
        rows.append([human.px, human.py, human.gx, human.gy,
                     human.v_pref, human.radius])
    array = np.asarray(rows, dtype=np.float64).round(9)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def collect(scene, cases):
    scene_id, humans, generator, radius, width = scene
    env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", humans, generator,
                          radius, width, 25, True)
    out = {}
    for case in cases:
        env.reset(options={"test_case": case})
        out[case] = layout_hash(env)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = {"scenes": {}, "splits": {}, "checks": {}}

    for scene in (DEV_CIRCLE, DEV_SQUARE, *[tuple(s) for s in SCENES]):
        registry["scenes"][scene[0]] = {
            "scene_id": scene[0], "human_num": scene[1], "scenario": scene[2],
            "circle_radius": scene[3], "square_width": scene[4],
            "config_sha256_16": scene_hash(scene)}

    plan = [
        ("debug_circle", DEV_CIRCLE, DEBUG),
        ("debug_square", DEV_SQUARE, DEBUG),
        ("D1_circle", DEV_CIRCLE, D1_CIRCLE),
        ("D1_square", DEV_SQUARE, D1_SQUARE),
        ("D2_circle", DEV_CIRCLE, D2_CIRCLE),
        ("D2_square", DEV_SQUARE, D2_SQUARE),
    ] + [(f"T_{scene[0]}", tuple(scene), T_CASES) for scene in SCENES] \
      + [(f"L_{scene[0]}", tuple(scene), L_CASES) for scene in SCENES]

    for name, scene, cases in plan:
        hashes = collect(scene, cases)
        registry["splits"][name] = {
            "scene_id": scene[0], "scene_config_sha256_16": scene_hash(scene),
            "cases": cases, "layout_hashes": hashes,
            "distinct_layouts": len(set(hashes.values())), "n": len(cases)}
        print(f"{name:22s} {len(cases):4d} 个 case -> "
              f"{len(set(hashes.values())):4d} 个不同布局"
              + ("" if len(set(hashes.values())) == len(cases) else "   *** 有重复 ***"))

    # Overlap between the splits that must stay separate.
    def keyset(name):
        s = registry["splits"][name]
        return {(s["scene_config_sha256_16"], h) for h in s["layout_hashes"].values()}

    dev_debug = (keyset("debug_circle") | keyset("debug_square")
                 | set().union(*[keyset(f"L_{s[0]}") for s in SCENES]))
    d1 = keyset("D1_circle") | keyset("D1_square")
    d2 = keyset("D2_circle") | keyset("D2_square")
    t = set().union(*[keyset(f"T_{s[0]}") for s in SCENES])

    checks = {
        "D1 vs D2": len(d1 & d2),
        "D1 vs debug": len(d1 & dev_debug),
        "D2 vs debug": len(d2 & dev_debug),
        "D1 vs T": len(d1 & t),
        "D2 vs T": len(d2 & t),
        "debug vs T": len(dev_debug & t),
    }
    registry["checks"] = checks
    if any(checks.values()):
        raise SystemExit(f"overlapping layout splits: {checks}")
    print("\n交集检查（都应为 0）：")
    for label, count in checks.items():
        print(f"  {label:14s} {count}")

    # The order also warns about generator wrap-around: the same layout coming
    # back under a different id.  Reported per split above; summarised here.
    duplicated = {name: s["n"] - s["distinct_layouts"]
                  for name, s in registry["splits"].items()
                  if s["n"] != s["distinct_layouts"]}
    registry["duplicate_layouts"] = duplicated
    if duplicated:
        raise SystemExit(f"duplicate layouts: {duplicated}")
    print(f"\n重复布局：{duplicated if duplicated else '无'}")

    # Hard gate: no split may reuse a layout any earlier experiment consumed.
    history = historical_case_ids()
    registry["historical_case_count"] = len(history)
    clashes = {}
    for name, entry in registry["splits"].items():
        if name.startswith("debug"):
            continue          # the debug split is deliberately already-seen
        overlap = sorted(set(entry["cases"]) & history)
        if overlap:
            clashes[name] = overlap[:10]
    registry["historical_clashes"] = clashes
    print(f"\n历史占用 case {len(history)} 个（扫描全部结果文件重建）")
    if clashes:
        print("*** 与历史实验重叠，拒绝登记：")
        for name, sample in clashes.items():
            print(f"    {name}: {sample} ...")
        raise SystemExit(1)
    print("与历史实验无重叠")

    target = OUT / "case_registry.json"
    target.write_text(json.dumps(registry, indent=1, ensure_ascii=False))
    print(f"\n登记表 -> {target}")
    total = sum(s["n"] for n, s in registry["splits"].items() if n.startswith(("D1", "D2")))
    print(f"开发布局合计 {total}（D1 40 + D2 100，各按 circle/square 对半）")


def verify_existing():
    registry = json.loads((OUT / "case_registry.json").read_text())
    history = historical_case_ids()
    groups = {name: set() for name in ("D1", "D2", "T", "debug")}
    for name, entry in registry["splits"].items():
        if not name.startswith("debug") and set(entry["cases"]) & history:
            raise RuntimeError(f"historical case overlap: {name}")
        scene = registry["scenes"][entry["scene_id"]]
        definition = (scene["scene_id"], scene["human_num"], scene["scenario"],
                      scene["circle_radius"], scene["square_width"])
        hashes = {str(case): value for case, value in collect(definition, entry["cases"]).items()}
        if hashes != entry["layout_hashes"] or len(set(hashes.values())) != entry["n"]:
            raise RuntimeError(f"layout changed or duplicated: {name}")
        if scene_hash(definition) != entry["scene_config_sha256_16"]:
            raise RuntimeError(f"scene configuration hash mismatch: {name}")
        group = name.split("_")[0]
        groups[group if group in groups else "debug"].update(hashes.values())
    for a, va in groups.items():
        for b, vb in groups.items():
            if a < b and va & vb:
                raise RuntimeError(f"actual layout overlap independent of scene label: {a}, {b}")
    print("Existing registry verified against generated layouts and historical records; not regenerated")


if __name__ == "__main__":
    verify_existing() if "--verify-existing" in sys.argv else main()
