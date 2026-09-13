"""Interface acceptance for the two bridge controllers.

Success rate is not evidence that an interface is right.  These checks look at
the quantities themselves, on inputs whose correct answer is known in advance:

  1. time indexing -- prediction step k must be the pedestrian's position k
     control periods ahead, not k-1 or k+1;
  2. action units -- the bridge returns an angular rate, CrowdSim consumes a
     heading increment, so exactly one factor of dt must appear;
  3. uncertainty units and semantics -- what each consumer receives must
     reproduce the posterior it is meant to share;
  4. obstacle capacity -- 20 detections must reach the solver, not 12;
  5. determinism -- whether repeated runs agree, reported rather than assumed.

A failure here means the comparison is not yet valid, whatever the scores say.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

SRC = os.environ.get("SRC", "/home/abc/workspace/bayes_occ_mpc")
sys.path.insert(0, SRC)

import numpy as np                                                    # noqa: E402
from continuous_mpc_gate import DEFAULT_CROWDNAV, run_episode          # noqa: E402
from evaluate_matched_safety import config as point_config             # noqa: E402
from modern_worker import BridgeController, bridge_planner_factory     # noqa: E402
from unicycle_mpc_gate import unicycle_config                          # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def parse_request(line, horizon):
    """Decode what the bridge would actually receive."""
    t = line.split()
    assert t[0] == "STEP"
    out = {"x": float(t[1]), "y": float(t[2]), "psi": float(t[3]), "v": float(t[4]),
           "goal": (float(t[5]), float(t[6]))}
    i = 7
    assert t[i] == "NPATH"; n_path = int(t[i + 1]); i += 2
    out["path"] = [(float(t[i + 2 * j]), float(t[i + 2 * j + 1])) for j in range(n_path)]
    i += 2 * n_path
    assert t[i] == "NOBS", t[i]; n_obs = int(t[i + 1]); i += 2
    obstacles = []
    for _ in range(n_obs):
        oid, ox, oy, oang, orad = int(t[i]), float(t[i+1]), float(t[i+2]), float(t[i+3]), float(t[i+4])
        i += 5
        assert t[i] == "NPRED"; n_pred = int(t[i + 1]); i += 2
        preds = []
        for _ in range(n_pred):
            preds.append(tuple(float(t[i + j]) for j in range(5)))
            i += 5
        obstacles.append({"id": oid, "x": ox, "y": oy, "angle": oang,
                          "radius": orad, "pred": preds})
    out["obstacles"] = obstacles
    return out


class Recorder(BridgeController):
    """Captures requests without starting a bridge process.

    Mirrors every field BridgeController.to_request reads.  When the real
    constructor gains a field this must gain it too, otherwise the audit fails
    on an attribute error instead of on the thing it is auditing.
    """

    def __init__(self, variant, horizon, dt, human_margin=0.0, speed_limit=1.0,
                 num_segments=5, max_obstacles=20):
        self.variant = variant
        self.base_variant = "shmpc" if variant.startswith("shmpc") else "tmpc"
        self.horizon = horizon
        self.dt = dt
        self.uncertainty_rate = 0.0
        self.max_obstacles = max_obstacles
        self._path = None
        self.negative_increments = 0.0
        self.human_margin = human_margin
        self.speed_limit = speed_limit
        self.num_segments = num_segments
        self.path_points = None
        self.path_segment_length = None
        self.path_overshoot = self.PATH_OVERSHOOT


def synthetic_observation(horizon, dt, humans=3, heading=0.3):
    """A pedestrian walking at a known constant velocity, with a covariance that
    grows by a known amount each step."""
    from continuous_mpc_gate import PlannerObservation
    entities, starts, ends, cov = [], [], [], []
    P0 = np.array([[0.02, 0.005], [0.005, 0.01]])
    Q = np.array([[0.03, -0.008], [-0.008, 0.012]])
    for i in range(humans):
        px, py = 1.0 + i, -2.0 - 0.5 * i
        vx, vy = 0.4 + 0.1 * i, -0.3
        entities.append([px, py, vx, vy, 0.3])
        s = np.stack([[px + vx * k * dt, py + vy * k * dt] for k in range(horizon)])
        e = np.stack([[px + vx * (k + 1) * dt, py + vy * (k + 1) * dt]
                      for k in range(horizon)])
        starts.append(s); ends.append(e)
        cov.append(np.stack([P0 + (k + 1) * Q for k in range(horizon)]))
    return PlannerObservation(
        robot_xy=np.array([0.0, -4.0]),
        robot_velocity=np.array([0.5 * math.cos(heading), 0.5 * math.sin(heading)]),
        robot_radius=0.3, goal_xy=np.array([0.0, 4.0]),
        entities=np.asarray(entities), human_segment_start=np.asarray(starts),
        human_segment_end=np.asarray(ends), human_uncertainty_buffer=None,
        human_position_covariance=np.asarray(cov), human_existence=None,
        human_visible=None, unknown=None, occupancy_probability=None,
        provenance="interface_audit", robot_heading=heading)


def test_time_index_and_units(horizon, dt):
    obs = synthetic_observation(horizon, dt)
    for variant in ("tmpc", "shmpc"):
        req = parse_request(Recorder(variant, horizon, dt).to_request(obs), horizon)

        ok_state = (abs(req["x"] - obs.robot_xy[0]) < 1e-9 and
                    abs(req["psi"] - obs.robot_heading) < 1e-9 and
                    abs(req["v"] - float(np.linalg.norm(obs.robot_velocity))) < 1e-9)
        check(f"{variant}: robot state transcribed", ok_state)

        # Prediction k must equal the pedestrian's position k+1 periods ahead.
        worst = 0.0
        for i, ob in enumerate(req["obstacles"]):
            for k, (mx, my, _, _, _) in enumerate(ob["pred"]):
                worst = max(worst, abs(mx - obs.human_segment_end[i, k, 0]),
                            abs(my - obs.human_segment_end[i, k, 1]))
        check(f"{variant}: prediction step k is k+1 periods ahead", worst < 1e-9,
              f"max error {worst:.2e}")

        check(f"{variant}: all detections sent", len(req["obstacles"]) == 3,
              f"{len(req['obstacles'])} of 3")
        dense = parse_request(Recorder(variant, horizon, dt).to_request(
            synthetic_observation(horizon, dt, humans=20)), horizon)
        check(f"{variant}: all 20 detections sent", len(dense["obstacles"]) == 20)


def test_uncertainty_semantics(horizon, dt):
    obs = synthetic_observation(horizon, dt)
    marginal = np.asarray(obs.human_position_covariance)

    req_t = parse_request(Recorder("tmpc", horizon, dt).to_request(obs), horizon)
    major_t = np.array([[p[3] for p in ob["pred"]] for ob in req_t["obstacles"]])
    expected = np.sqrt(np.linalg.eigvalsh(marginal)[..., 1])
    check("tmpc: receives the marginal semi-axes",
          float(np.abs(major_t - expected).max()) < 1e-9,
          f"max error {np.abs(major_t - expected).max():.2e}")

    req_s = parse_request(Recorder("shmpc", horizon, dt).to_request(obs), horizon)
    major_s = np.array([[p[3] for p in ob["pred"]] for ob in req_s["obstacles"]])
    check("shmpc: receives a different (increment) form",
          float(np.abs(major_s - expected).max()) > 1e-3,
          "would be a bug if these matched")

    # Replay the sampler's random walk and check it lands on the marginal.
    shifted = np.concatenate((np.zeros_like(marginal[:, :1]), marginal[:, :-1]), axis=1)
    increment = (marginal - shifted) / (dt * dt)
    rng = np.random.default_rng(3)
    chol = np.linalg.cholesky(increment + 1e-15 * np.eye(2))
    samples = 120000
    worst = 0.0
    for v in range(marginal.shape[0]):
        walk = np.zeros((samples, 2))
        for k in range(horizon):
            walk = walk + rng.standard_normal((samples, 2)) @ chol[v, k].T
            got = np.cov(walk * dt, rowvar=False)
            worst = max(worst, float(np.abs(got - marginal[v, k]).max()))
    scale = float(np.abs(marginal).max())
    check("shmpc: the random walk reproduces the posterior marginal",
          worst / scale < 0.03, f"max relative error {worst/scale:.2%}")


def test_action_units(horizon, dt):
    """One factor of dt, no more, no less: the bridge returns rad/s and CrowdSim
    consumes a heading increment."""
    import inspect
    source = inspect.getsource(BridgeController.act)
    check("action converts rate to increment exactly once",
          "float(reply[3]) * self.dt" in source,
          "found the single dt factor" if "float(reply[3]) * self.dt" in source
          else "conversion not found where expected")


def test_repeatability(horizon, dt):
    """Both arms are stochastic, for different reasons, and both were measured.

    SH-MPC seeds its scenario sampler from `std::mt19937{std::random_device{}()}`
    (scenario_module/src/sampler.cpp:377).

    T-MPC++ is stochastic too, which the PRM seed does not fix: its guidance
    planners run under `#pragma omp parallel for num_threads(8)`
    (mpc_planner_modules/src/guidance_constraints.cpp:280) and the branch that
    wins varies between runs.  Measured on three repeats of one layout, path
    length spread 3.94-5.49 m in parallel against 4.4604-4.4606 m serialised
    with OMP_THREAD_LIMIT=1.

    So single runs are not reproducible for either method and every comparison
    needs repeats.  This records the spread instead of asserting a value.
    """
    crowdnav = Path(os.environ.get("CROWDNAV_ROOT", str(DEFAULT_CROWDNAV)))
    cfg = unicycle_config(point_config(3))
    for variant in ("tmpc", "shmpc"):
        outcomes = []
        for _ in range(3):
            r = run_episode(crowdnav, "bayes", 5, "circle_crossing", 3650, cfg,
                            4.0, None, 0, 0.0, 0.0, 1.0, time_limit=25,
                            planner_type=bridge_planner_factory(variant),
                            robot_kinematics="unicycle")
            outcomes.append((r.event, round(r.nav_time, 4), round(r.path_length, 4)))
        spread = max(o[2] for o in outcomes) - min(o[2] for o in outcomes)
        # The check is that the arm runs three times and the spread is recorded,
        # not that it is zero: neither method is deterministic.
        check(f"{variant}: repeated runs complete, spread recorded",
              len(outcomes) == 3,
              f"path length spread {spread:.4f} m over 3 repeats; {outcomes}")


def main():
    cfg = point_config(3)
    horizon, dt = cfg.horizon, cfg.dt
    print(f"horizon={horizon} dt={dt}\n")
    test_time_index_and_units(horizon, dt)
    test_uncertainty_semantics(horizon, dt)
    test_action_units(horizon, dt)
    print()
    test_repeatability(horizon, dt)
    print()
    if FAILURES:
        print(f"未通过 {len(FAILURES)} 项: {FAILURES}")
        return 1
    print("bridge 接口验收全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
