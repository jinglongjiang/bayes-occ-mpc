"""The refactor must change nothing at default settings."""
import importlib.util, sys, json
sys.path.insert(0, "/home/abc/workspace/bayes_occ_mpc")

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

which = sys.argv[1]
path = ("/tmp/claude-1000/cmg_before_ablation.py" if which == "before"
        else "/home/abc/workspace/bayes_occ_mpc/continuous_mpc_gate.py")
m = load(path, "continuous_mpc_gate")
from evaluate_matched_safety import config
from unicycle_mpc_gate import UnicycleCEMMPC
import modern_dynamics
out = []
for kin, planner in (("holonomic", m.ContinuousCEMMPC),
                     ("unicycle", UnicycleCEMMPC),
                     ("unicycle", modern_dynamics.ContinuousUnicycleMPC)):
    for case in (20000, 20001):
        r = m.run_episode(m.DEFAULT_CROWDNAV, "bayes", 5, "circle_crossing", case,
                          config(4), 4.0, None, 0, 0.0, 0.0, 1.0, time_limit=25,
                          planner_type=planner, robot_kinematics=kin)
        out.append([planner.__name__, case, r.event, round(r.nav_time, 12),
                    round(r.path_length, 12), round(r.min_clearance, 12),
                    r.solver_steps, r.observation_hash])
print(json.dumps(out))
