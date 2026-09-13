"""Check the bootstrap and Holm code against cases with known answers."""
import sys, numpy as np
sys.path.insert(0, "/home/abc/temp/modern")
import analysis

# 1. Holm: known worked example.
tests = {"a": {"p": 0.01}, "b": {"p": 0.04}, "c": {"p": 0.03}}
got = analysis.holm(tests)
# Sorted p: a=.01, c=.03, b=.04 with m=3.  Holm multiplies by (m - rank) and
# then enforces monotonicity, so b inherits c's 0.06 rather than its own 0.04.
expect = {"a": 0.03, "c": 0.06, "b": 0.06}
ok = all(abs(got[k] - v) < 1e-12 for k, v in expect.items())
print("Holm 校正:", {k: round(v, 4) for k, v in got.items()}, "->", "通过" if ok else "失败")

# 2. Bootstrap on an identical pair must give a zero difference and a CI at zero.
rng = np.random.default_rng(1)
layouts = {"s1": {i: {"cr": float(i % 3 == 0), "sr": 1.0, "time": 5.0} for i in range(100)}}
analysis.BOOTSTRAP = 2000
same = analysis.paired_bootstrap(layouts, layouts, "cr", rng)
print(f"相同两臂: 差值 {same['difference']:.6f} CI [{same['ci_low']:.4f},{same['ci_high']:.4f}] "
      f"p={same['p']:.3f} ->", "通过" if abs(same['difference']) < 1e-12
      and abs(same['ci_low']) < 1e-12 and abs(same['ci_high']) < 1e-12 else "失败")

# 3. A known constant difference must be recovered exactly, with a CI containing it.
shifted = {"s1": {i: {"cr": layouts["s1"][i]["cr"], "sr": 0.5, "time": 7.0} for i in range(100)}}
diff = analysis.paired_bootstrap(layouts, shifted, "time", rng)
print(f"已知差 -2.0: 得 {diff['difference']:.6f} CI [{diff['ci_low']:.4f},{diff['ci_high']:.4f}] ->",
      "通过" if abs(diff["difference"] + 2.0) < 1e-9 else "失败")

# 4. Macro must weight scenes equally, not episodes: a 1-layout scene counts as
#    much as a 99-layout scene.
uneven = {"small": {0: {"cr": 1.0}}, "big": {i: {"cr": 0.0} for i in range(99)}}
m = analysis.macro(uneven, "cr")
print(f"场景等权 (期望 0.5): {m:.4f} ->", "通过" if abs(m - 0.5) < 1e-12 else "失败")

# 5. The four audited outcomes must be mutually exclusive and sum to 1.
for row in ({"event": "reach_goal", "success_without_overlap": 1, "collision_union": 0, "nav_time": 9.0},
            {"event": "timeout", "success_without_overlap": 0, "collision_union": 0, "nav_time": 25.0},
            {"event": "timeout", "success_without_overlap": 0, "collision_union": 1, "nav_time": 25.0},
            {"event": "error", "success_without_overlap": 0, "collision_union": None, "nav_time": 25.0}):
    v = analysis.episode_values(row)
    total = v["sr"] + v["cr"] + v["tr"] + v["err"]
    print(f"  {row['event']:10s} SR{v['sr']:.0f} CR{v['cr']:.0f} TR{v['tr']:.0f} "
          f"ERR{v['err']:.0f} 未知CR{v['cr_unknown']:.0f} 合计{total:.0f} ->",
          "通过" if abs(total - 1.0) < 1e-12 else "失败")
