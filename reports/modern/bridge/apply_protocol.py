"""Bring the mpc_planner settings onto this project's registered protocol.

Every value here is fixed before any development run and is identical for
T-MPC++ and SH-MPC.  The upstream defaults it replaces are recorded alongside,
so a later reader can see exactly what was changed and why.
"""
import pathlib, re, sys

# key: (new value, upstream default, why)
CHANGES = {
    "N":                  (16,   30,   "4 s horizon at dt 0.25, matching the project's 16-step planner"),
    "integrator_step":    (0.25, 0.2,  "the project's control period"),
    "control_frequency":  (4,    20,   "1 / 0.25 s; the deadline every method is held to"),
    "max_obstacles":      (20,   12,   "the largest registered crowd; 12 would silently drop pedestrians"),
    "robot_radius":       (0.3,  0.325,"CrowdNav robot radius"),
    "obstacle_radius":    (0.3,  0.4,  "CrowdNav pedestrian radius"),
}
NESTED = {"reference_velocity": (1.0, 2.0, "the project's v_max")}

path = pathlib.Path(sys.argv[1])
text = path.read_text()
report = []
for key, (new, old, why) in CHANGES.items():
    pattern = re.compile(rf"^({re.escape(key)}:\s*)([-\d.]+)", re.M)
    match = pattern.search(text)
    if not match:
        raise SystemExit(f"未找到键 {key}")
    found = match.group(2)
    text = pattern.sub(lambda m, n=new: f"{m.group(1)}{n}", text, count=1)
    report.append(f"  {key:20s} {found:>6s} -> {new:<6}  ({why})")
for key, (new, old, why) in NESTED.items():
    pattern = re.compile(rf"^(\s+{re.escape(key)}:\s*)([-\d.]+)", re.M)
    match = pattern.search(text)
    if not match:
        raise SystemExit(f"未找到键 {key}")
    found = match.group(2)
    text = pattern.sub(lambda m, n=new: f"{m.group(1)}{n}", text, count=1)
    report.append(f"  {key:20s} {found:>6s} -> {new:<6}  ({why})")
path.write_text(text)
print("\n".join(report))
