"""How often can cross-pedestrian evidence actually be used?

The narrowed proposal updates an occluded pedestrian from a visible neighbour.
That requires an occluded pedestrian to have a visible neighbour close enough
for their motions to be coupled.  If the situation is rare, the mechanism's
reachable share of the risk is small no matter how well it is implemented, and
the power analysis says the closed-loop experiment cannot resolve it.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
import numpy as np
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env, _load_modules

RADII = (1.5, 2.5, 4.0)


def main():
    _, _, ActionXY, _, _ = _load_modules(DEFAULT_CROWDNAV)
    print(f"{'场景':16s} {'遮挡人次':>9s} " +
          " ".join(f"{'有可见邻居<'+str(r)+'m':>16s}" for r in RADII))
    for scene, n, gen, r_, w_ in (("baseline_circle", 5, "circle_crossing", 4.0, None),
                                  ("dense_circle", 10, "circle_crossing", 4.0, None),
                                  ("dense_square", 20, "square_crossing", None, 10.0)):
        env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", n, gen, r_, w_, 25, True)
        total, has = 0, {r: 0 for r in RADII}
        for case in range(12000, 12030):
            env.reset(options={"test_case": case})
            for _ in range(40):
                vis = set(env.occlusion.visible_ids)
                pos = np.array([[h.px, h.py] for h in env.humans])
                for i in range(len(env.humans)):
                    if i in vis:
                        continue
                    total += 1
                    for r in RADII:
                        if any(j in vis and np.linalg.norm(pos[i]-pos[j]) < r
                               for j in range(len(env.humans)) if j != i):
                            has[r] += 1
                _, _, a, b, _ = env.step(ActionXY(0.0, 0.0))
                if a or b:
                    break
        print(f"{scene:16s} {total:9d} " +
              " ".join(f"{100*has[r]/max(total,1):15.1f}%" for r in RADII))
    print("\n注：这只是『存在一个可见邻居』的上界。还要求两人运动确实耦合、")
    print("    且该耦合能从合法可见历史中识别出来，实际可用比例只会更低。")


if __name__ == "__main__":
    main()
