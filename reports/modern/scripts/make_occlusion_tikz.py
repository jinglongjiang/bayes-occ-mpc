"""Emit the observability figure as self-contained TikZ, from real episodes.

The manuscript source deliberately requires no external figure files, so this
writes TikZ coordinates rather than an image.  Every number below comes from the
simulator: the shadow polygons use the same tangent-line construction the
occlusion module uses to decide visibility, and the visible/occluded split is
the module's own.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")
import numpy as np
from continuous_mpc_gate import DEFAULT_CROWDNAV, build_env, _load_modules

PANELS = [("baseline\\_circle", 5, "circle_crossing", 4.0, None, 12003),
          ("dense\\_circle", 10, "circle_crossing", 4.0, None, 12001),
          ("dense\\_square", 20, "square_crossing", None, 10.0, 12002)]
STEPS, HALF = 6, 6.2          # panel half-width in metres


def shadow(robot, h, far=30.0):
    dx, dy = h["px"] - robot[0], h["py"] - robot[1]
    d = float(np.hypot(dx, dy))
    if d <= h["radius"] or d < 1e-6:
        return None
    a = np.arctan2(dy, dx)
    th = np.arcsin(np.clip(h["radius"] / d, -1.0, 1.0))
    t = h["radius"] / max(np.tan(th), 1e-6)
    p1 = (robot[0] + t*np.cos(a-th), robot[1] + t*np.sin(a-th))
    p2 = (robot[0] + t*np.cos(a+th), robot[1] + t*np.sin(a+th))
    def ext(p):
        v = np.array(p) - robot
        n = np.hypot(*v)
        return tuple(np.array(p) + v / n * far) if n > 1e-9 else p
    return [p1, p2, ext(p2), ext(p1)]


def collect():
    _, _, ActionXY, _, _ = _load_modules(DEFAULT_CROWDNAV)
    out = []
    for title, n, gen, radius, width, case in PANELS:
        env, _, _ = build_env(DEFAULT_CROWDNAV, "bayes", n, gen, radius, width, 25, True)
        env.reset(options={"test_case": case})
        for _ in range(STEPS):
            env.step(ActionXY(0.0, 0.0))
        robot = np.array([env.robot.px, env.robot.py])
        goal = np.array([env.robot.gx, env.robot.gy])
        fov = float(getattr(env.occlusion, "fov_radius", 5.0))
        visible = set(env.occlusion.visible_ids)
        humans = [{"id": i, "px": h.px, "py": h.py, "radius": h.radius}
                  for i, h in enumerate(env.humans)]
        cat, counts = [], {"seen": 0, "shadow": 0, "range": 0}
        for h in humans:
            d = float(np.hypot(h["px"]-robot[0], h["py"]-robot[1]))
            k = "seen" if h["id"] in visible else ("range" if d > fov else "shadow")
            counts[k] += 1
            cat.append((h, k))
        polys = [shadow(robot, humans[k]) for k in np.argsort(
            [np.hypot(h["px"]-robot[0], h["py"]-robot[1]) for h in humans])]
        # Frame on the robot, biased towards the goal so the task is visible.
        centre = robot + (goal - robot) * 0.42
        out.append(dict(title=title, n=n, robot=robot, goal=goal, fov=fov,
                        centre=centre, cat=cat, counts=counts,
                        polys=[p for p in polys if p],
                        robot_radius=float(env.robot.radius)))
    return out


def panel_tikz(d):
    cx, cy = d["centre"]
    def xy(p):
        return f"({p[0]-cx:.3f},{p[1]-cy:.3f})"
    L = []
    L.append("\\begin{tikzpicture}[scale=0.36,every node/.style={font=\\scriptsize}]")
    L.append(f"\\clip (-{HALF},-{HALF}) rectangle ({HALF},{HALF});")
    L.append(f"\\draw[fill=white,draw=black!55] (-{HALF},-{HALF}) rectangle ({HALF},{HALF});")
    for poly in d["polys"]:
        pts = " -- ".join(xy(p) for p in poly)
        L.append(f"\\fill[black!30,opacity=0.32] {pts} -- cycle;")
    L.append(f"\\draw[dashed,thick,blue!65!black] {xy(d['robot'])} circle ({d['fov']:.2f});")
    L.append(f"\\draw[densely dotted,black!55] {xy(d['robot'])} -- {xy(d['goal'])};")
    style = {"seen": "green!55!black", "shadow": "red!75!black", "range": "black!35"}
    for h, k in d["cat"]:
        L.append(f"\\filldraw[fill={style[k]},draw=black,line width=0.3pt] "
                 f"{xy((h['px'],h['py']))} circle ({h['radius']:.2f});")
    L.append(f"\\filldraw[fill=blue!70!black,draw=black,line width=0.4pt] "
             f"{xy(d['robot'])} circle ({d['robot_radius']:.2f});")
    L.append(f"\\node[star,star points=5,star point ratio=2.2,inner sep=1.6pt,"
             f"fill=orange!85!black,draw=black,line width=0.3pt] at {xy(d['goal'])} {{}};")
    L.append("\\end{tikzpicture}")
    return "\n".join(L)


def main():
    data = collect()
    parts = ["\\begin{figure*}[tb]", "\\centering"]
    for i, d in enumerate(data):
        c = d["counts"]
        parts.append("\\begin{minipage}[t]{0.32\\linewidth}\\centering")
        parts.append(panel_tikz(d))
        parts.append(f"\\\\[2pt]{{\\footnotesize\\textbf{{{d['title']}}} ({d['n']} pedestrians)\\\\"
                     f"visible {c['seen']}\\quad occluded {c['shadow']}\\quad "
                     f"out of range {c['range']}}}")
        parts.append("\\end{minipage}" + ("\\hfill" if i < len(data)-1 else ""))
    parts.append("\\\\[6pt]")
    parts.append("{\\footnotesize\\textcolor{blue!70!black}{$\\bullet$}~robot\\quad"
                 "\\textcolor{orange!85!black}{$\\star$}~goal\\quad"
                 "\\textcolor{green!55!black}{$\\bullet$}~visible\\quad"
                 "\\textcolor{red!75!black}{$\\bullet$}~occluded by another pedestrian\\quad"
                 "\\textcolor{black!35}{$\\bullet$}~beyond sensing range\\quad"
                 "\\textcolor{black!30}{$\\blacksquare$}~cast shadow\\quad"
                 "\\textcolor{blue!65!black}{-\\,-}~5\\,m sensing radius}")
    parts.append("""\\caption{What limits observability in this benchmark, drawn from three
executed episodes. Shadows are the tangent-line polygons the simulator itself
uses to decide visibility, and the visible/occluded split is the simulator's.
Two distinct causes are separated: a pedestrian hidden behind another body
(red) and one simply farther away than the 5\\,m sensor reaches (grey). The
scenes contain no walls or static obstacles, so every occluder is another
pedestrian. Inter-agent occlusion therefore appears only once the crowd is dense
enough for people to line up with each other; in the five-person scene the
missing pedestrians are all out of range rather than occluded.}""")
    parts.append("\\label{fig:observability}")
    parts.append("\\end{figure*}")
    body = "\n".join(parts) + "\n"
    open("/home/abc/temp/modern/occlusion_figure.tex", "w").write(body)
    print("counts:", [d["counts"] for d in data])
    print("-> /home/abc/temp/modern/occlusion_figure.tex")


if __name__ == "__main__":
    main()
