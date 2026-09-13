from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mechanism_continuous import (Brake, CheckedPlanner, SingleMode, cfg,
                                  check_prefix, execution_acceptance, validate_row)
from continuous_mpc_gate import PlannerObservation


def observation(speed):
    return PlannerObservation(
        robot_xy=np.zeros(2), robot_velocity=np.array([speed, 0.]), robot_radius=.3,
        goal_xy=np.array([5., 0.]), entities=np.empty((0, 5)),
        human_segment_start=None, human_segment_end=None, human_uncertainty_buffer=None,
        human_position_covariance=None, human_existence=None, human_visible=None,
        unknown=None, occupancy_probability=None, provenance="test", robot_heading=0.)


def test_execution_matches_rollout_and_independent_integral():
    assert len(execution_acceptance()) == 6


@pytest.mark.parametrize("speed", [0., .3, .8, 1.])
def test_stop_seed_respects_deceleration(speed):
    planner = Brake(cfg())
    planner.fallback_calls = 0
    obs = observation(speed)
    seeds = planner._seed_trajectories(obs)
    index = planner._degraded_choice(np.zeros(len(seeds)), np.zeros(len(seeds)), seeds, obs)
    assert index == 1
    np.testing.assert_allclose(seeds[index, 0], [max(speed-.5, 0.), 0.], atol=1e-12, rtol=0.)


def test_single_mode_keeps_route_candidates_and_cost_budget():
    obs = observation(.6)
    current, single = CheckedPlanner(cfg()), SingleMode(cfg())
    original_seeds = current._seed_trajectories(obs)
    routes = current._route_seeds(obs)
    np.testing.assert_array_equal(single._seed_trajectories(obs),
                                  np.concatenate((original_seeds, routes)))
    assert single._route_seeds(obs).shape == (0, cfg().horizon, 2)
    for planner in (current, single):
        planner.plan(obs, 91827)
        assert planner.evaluations == 2048


def test_brake_is_identical_before_fallback():
    obs = observation(.6)
    current, brake = CheckedPlanner(cfg()), Brake(cfg())
    for seed in (71, 72, 73):
        a, _ = current.plan(obs, seed)
        b, _ = brake.plan(obs, seed)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(current.last_controls, brake.last_controls)
        assert current.last_diagnostics == brake.last_diagnostics


def test_prefix_rejects_changes_before_trigger():
    base = dict(layout_hash="same", audited_event="reach_goal", steps=[dict(
        feasibility_class=0, observation_hash="a", before=[0., 0.], action_a=.5,
        action_b=0., x=.0625, y=0., theta=0., speed=.5, clearance=1.)])
    other = dict(base, steps=[dict(base["steps"][0], x=.07)])
    with pytest.raises(AssertionError):
        check_prefix(base, other)


def test_validator_rejects_missing_or_wrong_executor():
    row = dict(arm="current", scene_id="circle5", case_id=20100,
               execution_model="discrete_unicycle")
    with pytest.raises(AssertionError):
        validate_row(row, "current", ("circle5", 5), 20100, "hash")
    del row["execution_model"]
    with pytest.raises(KeyError):
        validate_row(row, "current", ("circle5", 5), 20100, "hash")
