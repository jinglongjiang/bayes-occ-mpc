from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.stats import ncx2

import continuous_mpc_gate as core
from bayesian_rfs import RFSConfig, BernoulliTrack
from spatial_belief import (DetectionSupport, ParticleState, SpatialAdapter,
                            SpatialBelief, missed_update, moments, normals)
from spatial_risk_mpc import (SpatialRiskMPC, swept_statistics,
                              swept_statistics_torch)
from unicycle_mpc_gate import unicycle_config


def grid():
    axis = np.arange(-3., 3.01, .25)
    mesh = np.meshgrid(axis, axis)
    sensor = np.where(mesh[0] < 0., 0., .5)
    return sensor, mesh


def observation(samples, horizon=2, existence=1.):
    count = len(samples)
    mean = samples.mean(axis=0)
    obs = core.PlannerObservation(
        robot_xy=np.zeros(2), robot_velocity=np.zeros(2), robot_radius=.3,
        goal_xy=np.array([0., 5.]), entities=np.array([[*mean[:2], *mean[2:], .3]]),
        human_segment_start=np.broadcast_to(mean[:2], (1, horizon, 2)).copy(),
        human_segment_end=np.broadcast_to(mean[:2], (1, horizon, 2)).copy(),
        human_uncertainty_buffer=None, human_position_covariance=np.zeros((1, horizon, 2, 2)),
        human_existence=np.array([existence]), human_visible=np.array([False]),
        unknown=None, occupancy_probability=None, provenance="synthetic", robot_heading=0.)
    times = np.arange(horizon + 1) * .25
    states = np.broadcast_to(samples[None, :, None, :], (1, count, horizon + 1, 4)).copy()
    states[0, :, :, :2] += samples[:, None, 2:] * times[None, :, None]
    obs.posterior_states = states
    obs.posterior_accelerations = np.zeros((1, count, horizon, 2))
    obs.posterior_weights = np.full((1, count), 1. / count)
    obs.posterior_count = count
    return obs


def evaluate(obs, subdivisions=8):
    from spatial_risk_mpc import human_path_nodes
    cfg = unicycle_config(core.MPCConfig(
        horizon=obs.posterior_states.shape[2] - 1, human_margin=.1,
        near_chance_limit=.3, chance_limit=.3))
    planner = SpatialRiskMPC(cfg, subdivisions)
    planner._human_nodes = human_path_nodes(obs, cfg.dt, subdivisions)
    planner._human_bound = np.linalg.norm(obs.posterior_accelerations, axis=-1)
    _, controls, positions = planner._rollout(np.zeros((1, cfg.horizon, 2)), obs)
    return planner, controls, positions


def test_missed_update_conditional_and_existence():
    r, weights = missed_update(.99, np.array([.5, .5]), np.array([.98, 0.]))
    np.testing.assert_allclose(weights, [1 / 51, 50 / 51])
    np.testing.assert_allclose(r, .99 * .51 / (.01 + .99 * .51))
    r2, w2 = missed_update(.99, np.array([.5, .5]), np.array([.98, .98]))
    np.testing.assert_allclose(w2, [.5, .5])
    assert r2 < r


def test_no_eligibility_is_no_negative_evidence():
    r, w = missed_update(.9, np.array([.3, .7]), np.zeros(2))
    assert r == .9
    np.testing.assert_array_equal(w, [.3, .7])


def test_support_uses_only_observed_free_space_as_negative_evidence():
    sensor, mesh = grid()
    a = DetectionSupport(sensor, mesh)
    b = DetectionSupport(np.where(sensor == 0., 1., sensor), mesh)
    xy = np.array([[-1., 0.], [1., 0.], [10., 0.]])
    np.testing.assert_array_equal(a.eligible(xy, .3), [True, False, False])
    np.testing.assert_array_equal(b.eligible(xy, .3), [False, False, False])
    assert a.eligible(np.array([[.04, 0.]]), .3)[0]
    # Grid centres, as in the simulator: a .1m centre has no body cell at
    # x=-.25m, although a continuous silhouette would cross x=0.
    assert not a.eligible(np.array([[.1, 0.]]), .3)[0]


def test_visible_occluder_cells_do_not_disprove_hidden_target():
    sensor, mesh = grid()
    # A visible occupied return is not free-space evidence.  This protects a
    # hidden hypothesis whose footprint overlaps the visible occluder.
    sensor[(mesh[0] + 1.) ** 2 + mesh[1] ** 2 <= .3 ** 2] = 1.
    support = DetectionSupport(sensor, mesh)
    assert not support.eligible(np.array([[-1., 0.]]), .05)[0]
    assert support.observable(np.array([[-1., 0.]]), .05)[0]


def test_joint_negative_evidence_updates_velocity():
    sensor, mesh = grid()
    cfg = replace(RFSConfig(), acceleration_std=0., survival_probability=1.)
    belief = SpatialBelief(cfg, count=8)
    states = np.tile([[-1., 0., -1., 0.], [1., 0., 1., 0.]], (4, 1))
    belief.tracks[1] = BernoulliTrack(1, np.zeros(4), np.eye(4), .99)
    belief.particles[1] = ParticleState(states, np.full(8, .125), np.zeros(8, dtype=int))
    belief._last_time = 0.
    belief.update([], sensor, mesh, .25)
    assert belief.tracks[1].mean[0] > 1.
    assert belief.tracks[1].mean[2] > .9


def test_hard_deletion_requires_registered_streak():
    sensor, mesh = grid()
    cfg = replace(RFSConfig(), acceleration_std=0., survival_probability=1.)
    belief = SpatialBelief(cfg, count=8, hard_streak=2)
    detection = dict(id=0, px=-1., py=0., vx=0., vy=0., radius=.3)
    belief.update([detection], sensor, mesh, 0.)
    belief.update([], sensor, mesh, .25)
    assert 0 in belief.tracks
    belief.update([], sensor, mesh, .5)
    assert not belief.tracks
    assert belief.diagnostics["hard_exhaustions"] == 1


def test_projection_does_not_modify_tracker(monkeypatch):
    sensor, mesh = grid()
    state = SimpleNamespace(self_state=SimpleNamespace(
        px=0., py=-2., vx=0., vy=0., gx=0., gy=2., radius=.3, theta=0.),
        provenance="test")
    detection = dict(id=0, px=-.1, py=0., vx=0., vy=0., radius=.3)
    env = SimpleNamespace(global_time=0.)
    monkeypatch.setattr(core, "policy_observation_frame", lambda _: (
        state, [detection] if env.global_time == 0. else [],
        {0} if env.global_time == 0. else set(), sensor, mesh, .25))
    c = SpatialAdapter("bayes", representation="projected", count=64)
    d = SpatialAdapter("bayes", representation="shape", count=64)
    for step in range(4):
        env.global_time = step * .25
        c.read(env)
        d.read(env)
        assert c.reported_ids == d.reported_ids
        for identifier in c.rfs.tracks:
            np.testing.assert_array_equal(c.rfs.particles[identifier].states,
                                          d.rfs.particles[identifier].states)
            np.testing.assert_array_equal(c.rfs.particles[identifier].weights,
                                          d.rfs.particles[identifier].weights)


def test_bimodal_gap_has_no_mean_obstacle_in_any_consumer():
    samples = np.tile([[-1.2, 0., 0., 0.], [1.2, 0., 0., 0.]], (32, 1))
    obs = observation(samples)
    planner, controls, positions = evaluate(obs)
    clearance = planner._human_clearance(controls, obs)
    assert np.isinf(clearance).all()
    hazard = planner._belief_collision_hazard(controls, obs, positions)
    np.testing.assert_array_equal(hazard, 0.)
    full, first = planner._combined_clearance(controls, obs, positions, clearance, None, hazard)
    assert full[0] >= 0. and first[0] >= 0.
    _, penalty, comfort = planner._evaluate(obs)
    assert not penalty.any() and not comfort.any()
    _, fallback = planner._physical_clearance(controls, obs, positions, clearance)
    assert fallback[0] == 0.


def test_crossing_between_endpoints_is_detected():
    # Endpoints are each 1m away, but the person crosses the stopped robot.
    obs = observation(np.tile([-1., 0., 8., 0.], (64, 1)), horizon=1)
    planner, controls, positions = evaluate(obs)
    np.testing.assert_allclose(planner._evaluate(obs)[0], 1.)
    full, first = planner._combined_clearance(
        controls, obs, positions, planner._human_clearance(controls, obs), None,
        planner._belief_collision_hazard(controls, obs, positions))
    assert full[0] < 0. and first[0] < 0.


def test_hidden_risk_remains_in_fallback():
    obs = observation(np.tile([0., 0., 0., 0.], (64, 1)))
    planner, controls, positions = evaluate(obs)
    _, first = planner._physical_clearance(
        controls, obs, positions, planner._human_clearance(controls, obs))
    np.testing.assert_allclose(first, -1.)


def test_gaussian_reference_against_analytic_disc_mass():
    samples = np.zeros((4096, 4))
    samples[:, :2] = [.8, 0.] + .3 * normals(4096, 2, 7)
    obs = observation(samples, horizon=1, existence=.8)
    planner, _, _ = evaluate(obs)
    expected = .8 * ncx2.cdf((.7 / .3) ** 2, 2, (.8 / .3) ** 2)
    assert abs(planner._evaluate(obs)[0][0, 0, 0] - expected) < .012


def test_sweep_subdivision_linear_case_is_invariant():
    obs = observation(np.tile([-1., .1, 8., 0.], (64, 1)), horizon=1)
    p8, _, _ = evaluate(obs, 8)
    p32, _, _ = evaluate(obs, 32)
    for a, b in zip(p8._evaluate(obs), p32._evaluate(obs)):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_gaussian_samples_match_moments():
    z = normals(64, 4, 5)
    mean, covariance = moments(z, np.full(64, 1. / 64))
    np.testing.assert_allclose(mean, 0., atol=1e-14)
    np.testing.assert_allclose(covariance, np.eye(4), atol=1e-12)


def test_cuda_swept_statistics_matches_cpu_reference():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(29)
    # The CPU broad phase is exact under the documented acceleration bound,
    # not for arbitrary unrelated polyline nodes.  Linear paths exercise the
    # same swept calculation while satisfying that contract with zero bound.
    fractions = np.linspace(0., 1., 5)
    robot_start = rng.normal(size=(7, 3, 1, 2))
    robot_delta = rng.normal(size=(7, 3, 1, 2))
    robot = robot_start + fractions[None, None, :, None] * robot_delta
    human_start = rng.normal(size=(2, 16, 3, 1, 2))
    human_delta = rng.normal(size=(2, 16, 3, 1, 2))
    human = human_start + fractions[None, None, None, :, None] * human_delta
    robot_bound = np.zeros((7, 3))
    human_bound = np.zeros((2, 16, 3))
    weights = rng.uniform(size=(2, 16))
    weights /= weights.sum(axis=1, keepdims=True)
    existence = np.array([.8, .95])
    radii = np.array([.6, .7])
    arguments = (robot, robot_bound, human, human_bound, weights,
                 existence, radii, .1, .25)
    cpu = swept_statistics(*arguments)
    gpu = swept_statistics_torch(*arguments, chunk_size=3)
    np.testing.assert_allclose(gpu[0], cpu[0], atol=2e-7, rtol=0.)
    np.testing.assert_allclose(gpu[1], cpu[1], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(gpu[2], cpu[2], atol=2e-6, rtol=2e-5)
