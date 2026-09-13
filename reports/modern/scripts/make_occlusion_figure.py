"""Draw what 'occlusion' actually means in this benchmark, from a real episode.

Not a schematic: the scene, the pedestrians, the shadow polygons and the
visible/occluded split are taken from the simulator's own occlusion module, using
the same tangent-line construction it uses to decide visibility.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env, _load_modules


def shadow_polygon(robot, h, extent):
    dx, dy = h["px"] - robot[0], h["py"] - robot[1]
    d = float(np.hypot(dx, dy))
    if d <= h["radius"] or d < 1e-6:
        return None
    alpha = np.arctan2(dy, dx)
    theta = np.arcsin(np.clip(h["radius"] / d, -1.0, 1.0))
    t = h["radius"] / max(np.tan(theta), 1e-6)
    x1, y1 = robot[0] + t*np.cos(alpha-theta), robot[1] + t*np.sin(alpha-theta)
    x2, y2 = robot[0] + t*np.cos(alpha+theta), robot[1] + t*np.sin(alpha+theta)
    far = 2.5 * extent
    def extend(xa, ya, xt):
        return ya + (xt - xa) * (ya - robot[1]) / (xa - robot[0]) if abs(xa - robot[0]) > 1e-9 else ya
    x3 = robot[0] - far if x1 <= robot[0] else robot[0] + far
    x4 = robot[0] - far if x2 <= robot[0] else robot[0] + far
    y3 = extend(x1, y1, x3)
    y4 = extend(x2, y2, x4)
    return np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])


def panel(ax, env, title, steps):
    _, _, ActionXY, _, _ = _load_modules(DEFAULT_CROWDNAV)
    for _ in range(steps):
        env.step(ActionXY(0.0, 0.0))
    robot = np.array([env.robot.px, env.robot.py])
    occ = env.occlusion
    visible = set(occ.visible_ids)
    humans = [{"id": i, "px": h.px, "py": h.py, "radius": h.radius}
              for i, h in enumerate(env.humans)]
    extent = getattr(occ, "extent", 5.0)
    fov = getattr(occ, "fov_radius", 5.0)

    order = np.argsort([np.hypot(h["px"]-robot[0], h["py"]-robot[1]) for h in humans])
    for k in order:
        poly = shadow_polygon(robot, humans[k], extent)
        if poly is not None:
            ax.add_patch(Polygon(poly, closed=True, facecolor="0.55",
                                 alpha=0.30, edgecolor="none", zorder=1))

    ax.add_patch(Circle(robot, fov, fill=False, ls="--", lw=1.1,
                        ec="#1f77b4", zorder=3))
    # Two different reasons a pedestrian is missing must not be drawn the same
    # way: one is genuinely hidden behind another body, the other is simply
    # further away than the sensor reaches.
    counts = {"seen": 0, "shadow": 0, "range": 0}
    for h in humans:
        d = float(np.hypot(h["px"] - robot[0], h["py"] - robot[1]))
        if h["id"] in visible:
            colour, key = "#2ca02c", "seen"
        elif d > fov:
            colour, key = "#bbbbbb", "range"
        else:
            colour, key = "#d62728", "shadow"
        counts[key] += 1
        ax.add_patch(Circle((h["px"], h["py"]), h["radius"], zorder=4,
                            facecolor=colour, edgecolor="black", lw=0.7, alpha=0.95))
    ax.add_patch(Circle(robot, env.robot.radius, facecolor="#1f77b4",
                        edgecolor="black", lw=0.9, zorder=6))
    ax.plot([env.robot.gx], [env.robot.gy], marker="*", ms=15,
            color="#ff7f0e", mec="black", mew=0.6, zorder=6)
    ax.plot([robot[0], env.robot.gx], [robot[1], env.robot.gy],
            ls=":", lw=1.0, color="#666666", zorder=2)

    # Frame the robot, the goal and everything relevant between them.
    xs = [robot[0], env.robot.gx] + [h["px"] for h in humans]
    ys = [robot[1], env.robot.gy] + [h["py"] for h in humans]
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    lim = max(max(xs)-min(xs), max(ys)-min(ys))/2 + 1.0
    ax.set_xlim(cx-lim, cx+lim); ax.set_ylim(cy-lim, cy+lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nvisible {counts['seen']}    occluded {counts['shadow']}"
                 f"    out of range {counts['range']}", fontsize=10.5, pad=6)


def main():
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
    for ax, (scene, humans, gen, radius, width, case, steps) in zip(axes, [
            ("baseline_circle  (5 pedestrians)", 5, "circle_crossing", 4.0, None, 12003, 6),
            ("dense_circle  (10 pedestrians)", 10, "circle_crossing", 4.0, None, 12001, 6),
            ("dense_square  (20 pedestrians)", 20, "square_crossing", None, 10.0, 12002, 6)]):
        env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", humans, gen, radius, width, 25, True)
        env.reset(options={"test_case": case})
        panel(ax, env, scene, steps)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#1f77b4", mec="k", label="robot"),
        plt.Line2D([], [], marker="*", ls="", ms=13, mfc="#ff7f0e", mec="k", label="goal"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#2ca02c", mec="k", label="visible pedestrian"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#d62728", mec="k", label="occluded by another pedestrian"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#bbbbbb", mec="k", label="beyond sensing range"),
        plt.Line2D([], [], ls="--", color="#1f77b4", label="5 m sensing radius"),
        plt.Line2D([], [], marker="s", ls="", ms=10, mfc="0.55", mec="none",
                   alpha=.5, label="cast shadow"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=10)
    fig.suptitle("Limited-range sensing with inter-agent occlusion: "
                 "every occluder is another pedestrian, as the scenes contain "
                 "no walls or static obstacles", fontsize=12.5, y=0.98)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    out = "/home/abc/temp/modern/occlusion_figure.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
