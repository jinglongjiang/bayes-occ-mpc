import math
import json
import os
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.integrate import quad

from modern_dynamics import ramp_displacement, ContinuousUnicycleMPC, ContinuousExecutor
from modern_repair import JointBridgeController, empty_observation
from unicycle_mpc_gate import UnicycleConfig


class ContinuousDynamicsTests(unittest.TestCase):
    def test_deadline_has_no_extra_control_interval(self):
        from continuous_mpc_gate import DEFAULT_CROWDNAV
        sys.path.insert(0, str(DEFAULT_CROWDNAV))
        from crowd_sim.envs.utils.action import ActionRot
        for goal, expected in ((10., "timeout"), (.35, "reach_goal")):
            robot = SimpleNamespace(px=0., py=0., vx=0., vy=0., theta=0., radius=.3,
                                    visible=False, kinematics="unicycle")
            robot.get_position = lambda: (robot.px, robot.py)
            robot.get_goal_position = lambda: (goal, 0.)
            robot.get_full_state = lambda: (robot.px, robot.py)
            env = SimpleNamespace(robot=robot, humans=[], states=[], global_time=24.75,
                                  time_limit=25., time_step=.25)
            def step(command):
                robot.px += command.vx*.25
                robot.py += command.vy*.25
                env.global_time += .25
                return None, 0., False, False, {}
            env.step = step
            _, _, terminated, truncated, info = ContinuousExecutor(UnicycleConfig())(env, ActionRot(.5,0.))
            self.assertEqual(env.global_time,25.)
            self.assertEqual(info["event"],expected)
            self.assertTrue(terminated or truncated)

    def test_quadrature_matches_continuous_dynamics(self):
        rng = np.random.default_rng(729)
        for _ in range(100):
            theta, v0, v1, turn = rng.uniform(-math.pi, math.pi), rng.random(), rng.random(), rng.uniform(-.2, .2)
            result = ramp_displacement(theta, v0, v1, turn, .25)
            exact = [quad(lambda t: (v0 + (v1-v0)*t/.25) * f(theta + turn*t/.25), 0, .25)[0]
                     for f in (math.cos, math.sin)]
            np.testing.assert_allclose(result, exact, rtol=0., atol=1e-10)

    def test_zero_turn_and_start(self):
        np.testing.assert_allclose(ramp_displacement(0., 0., .5, 0., .25), [.0625, 0.], atol=1e-15)
        value = ramp_displacement(0., 1., np.ones(33), np.zeros(33), np.linspace(0, .25, 33))
        self.assertEqual(value.shape, (33, 2))
        np.testing.assert_allclose(value[0], [0., 0.])
        np.testing.assert_allclose(value[-1], [.25, 0.])

    def test_rollout_uses_endpoint_speed_with_continuous_motion(self):
        cfg = UnicycleConfig(horizon=3)
        planner = ContinuousUnicycleMPC(cfg)
        obs = empty_observation((0., 0.), (1., 1.), 0.)
        params, velocity, positions = planner._rollout(np.array([[[.5, .2], [1., -.2], [1., .0]]]), obs)
        xy = np.zeros(2); heading = 0.; speed = 0.
        for k, (v, turn) in enumerate(params[0]):
            xy += ramp_displacement(heading, speed, v, turn, cfg.dt)
            np.testing.assert_allclose(positions[0, k], xy, atol=1e-12)
            heading += turn; speed = v
        np.testing.assert_allclose(np.cumsum(velocity * cfg.dt, axis=1), positions)


class JointProtocolTests(unittest.TestCase):
    def test_startup_paths_are_idempotent_and_preserve_resolution_order(self):
        from modern_repair import BUILD, runtime_environment
        with patch.dict(os.environ, {"ROS_PACKAGE_PATH": "/extra:/extra::/last",
                                     "LD_LIBRARY_PATH": "/custom/lib:/custom/lib::/last/lib"}):
            runtime_environment()
            expected = {k: os.environ[k] for k in ("ROS_PACKAGE_PATH", "LD_LIBRARY_PATH")}
            self.assertEqual(expected["ROS_PACKAGE_PATH"].split(os.pathsep),
                [str(BUILD / "shmpc_n24/src"), str(BUILD / "tmpc/src"),
                 "/opt/ros/noetic/share", "/extra", "/last"])
            self.assertEqual(expected["LD_LIBRARY_PATH"].split(os.pathsep),
                ["/opt/ros/noetic/lib", "/custom/lib", "/last/lib"])
            for _ in range(1000):
                runtime_environment()
            self.assertEqual(expected, {k: os.environ[k] for k in expected})

    def controller(self):
        controller = object.__new__(JointBridgeController)
        controller.base_variant = 'shmpc'
        controller.variant = 'shmpc_repair'
        controller.horizon = 3; controller.dt = .25; controller.human_margin = .1
        controller.max_obstacles = 20; controller.speed_limit = 1.; controller.num_segments = 5
        controller.path_overshoot = 0.; controller._path = None; controller.sample_seed = 77
        controller.uncertainty_rate = 0.
        return controller

    def test_full_state_covariance_is_serialized_without_increment_conversion(self):
        obs = empty_observation((0., 0.), (1., 1.), 0.)
        obs.entities = np.array([[2., 2., 0., 0., .3]])
        obs.human_state_covariance = np.eye(4)[None]
        obs.human_position_covariance = np.repeat(np.eye(2)[None, None], 3, axis=1)
        controller = self.controller()
        msg = controller.to_request(obs)
        self.assertIn('JOINT_CV 77 0.55 1', msg)
        np.testing.assert_equal(controller._ellipses(obs)[0], np.ones((1,3)))
        self.assertEqual(controller._path[-1], (1., 1.))

    def test_missing_covariance_fails_instead_of_silently_falling_back(self):
        obs = empty_observation((0., 0.), (1., 1.), 0.)
        obs.entities = np.array([[2., 2., 0., 0., .3]])
        with self.assertRaisesRegex(ValueError, 'current position-velocity covariance'):
            self.controller().to_request(obs)

    def test_heading_wrap_does_not_jump_by_two_pi(self):
        controller = self.controller()
        obs = empty_observation((0., 0.), (1., 1.), -.05)
        first = controller.to_request(obs)
        obs.robot_heading = 2 * math.pi - .10
        second = controller.to_request(obs)
        self.assertAlmostEqual(float(first.split()[3]), -.05)
        self.assertAlmostEqual(float(second.split()[3]), -.10)
        self.assertAlmostEqual(obs.robot_heading, 2 * math.pi - .10)


@unittest.skipUnless(os.environ.get("MODERN_NATIVE_TESTS") == "1", "requires compiled native workspaces and ROS")
class NativeBuildTests(unittest.TestCase):
    def test_native_contracts_and_terminal_codegen(self):
        from modern_repair import BUILD, runtime_environment, settings
        runtime_environment()
        variants = [(f"{family}_repair_n{horizon}", constraints, horizon)
                    for horizon in (8, 16) for family, constraints in (("tmpc", 40), ("shmpc", 24))]
        for variant, constraints, horizon in variants:
            with self.subTest(variant=variant):
                path = BUILD / variant / "src/mpc_planner/mpc_planner_solver/acados/Solver/Solver.json"
                generated = json.loads(path.read_text())
                self.assertEqual(generated["cost"]["cost_type_e"], "EXTERNAL")
                np.testing.assert_array_equal(generated["solver_options"]["cost_scaling"],
                                              np.ones(generated["dims"]["N"] + 1))
                self.assertEqual(generated["dims"]["nh_e"], constraints)
                self.assertGreater(generated["dims"]["nbx_e"], 0)
                expected_fixed = generated["dims"]["nx"] - (1 if variant.startswith("shmpc") else 0)
                self.assertEqual(generated["dims"]["nbx_0"], expected_fixed)
                config, _ = settings(variant, "goal_track")
                controller = JointBridgeController(variant, horizon, .25, settings=str(config))
                try:
                    self.assertEqual(controller._exchange("CHECK_REFERENCE"), "REFERENCE_OK")
                    reply = controller._exchange("CHECK_WARMSTART").split()
                    self.assertEqual(reply[0], "WARMSTART_OK")
                    self.assertLess(float(reply[1]), 1e-10)
                    self.assertLessEqual(float(reply[2]), 2.)
                    obs = empty_observation((0., -4.), (0., 4.), math.pi / 2)
                    controller.act(obs)
                    fields = controller._exchange("TRAJECTORY").split()
                    self.assertEqual(fields[:2], ["TRAJECTORY", str(horizon)])
                    trajectory = np.asarray(fields[2:], dtype=float).reshape(horizon + 1, 7)
                    expected = obs.robot_xy + ramp_displacement(obs.robot_heading, 0.,
                        controller.last.speed, controller.last.turn, .25)
                    if controller.last.success:
                        np.testing.assert_allclose(trajectory[1, :2], expected, atol=1e-6)
                finally:
                    controller.close()


if __name__ == '__main__':
    unittest.main()
