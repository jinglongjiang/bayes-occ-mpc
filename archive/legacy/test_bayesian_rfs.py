import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import ncx2

from bayesian_rfs import BayesianRFSBelief, RFSConfig
from continuous_mpc_gate import ContinuousCEMMPC, MPCConfig, PlannerObservation


def mesh(size=20, resolution=0.25):
    coordinates = resolution * (np.arange(size) - size / 2 + 0.5)
    return np.meshgrid(coordinates, coordinates)


def detection(identifier=7, px=0.0, py=0.0, vx=0.5, vy=0.0):
    return {
        "id": identifier,
        "px": px,
        "py": py,
        "vx": vx,
        "vy": vy,
        "radius": 0.3,
    }


class BayesianRFSTest(unittest.TestCase):
    def setUp(self):
        self.mesh = mesh()
        self.unknown = np.full((20, 20), 0.5, dtype=np.float32)
        self.known = np.zeros((20, 20), dtype=np.float32)

    def test_measurement_then_occlusion_grows_uncertainty(self):
        belief = BayesianRFSBelief()
        belief.update([detection()], self.unknown, self.mesh, 0.0)
        initial = np.trace(belief.tracks[7].covariance[:2, :2])
        belief.update([], self.unknown, self.mesh, 0.25)
        predicted = np.trace(belief.tracks[7].covariance[:2, :2])
        self.assertGreater(predicted, initial)
        self.assertFalse(belief.tracks[7].visible)

    def test_birth_covariance_matches_measurement_noise(self):
        for position, velocity in ((0.,0.),(.05,.1),(.1,.2)):
            belief = BayesianRFSBelief(RFSConfig(position_measurement_std=position,
                                                velocity_measurement_std=velocity))
            belief.update([detection()],self.unknown,self.mesh,0.)
            np.testing.assert_array_equal(np.diag(belief.tracks[7].covariance),
                                          [position**2]*2+[velocity**2]*2)

    def test_visible_free_missed_detection_removes_track(self):
        belief = BayesianRFSBelief()
        belief.update([detection()], self.unknown, self.mesh, 0.0)
        belief.update([], self.known, self.mesh, 0.25)
        self.assertLess(belief.tracks[7].existence, 0.8)
        belief.update([], self.known, self.mesh, 0.5)
        self.assertNotIn(7, belief.tracks)

    def test_deleted_track_can_be_born_again_without_stale_state(self):
        belief = BayesianRFSBelief()
        belief.update([detection()], self.unknown, self.mesh, 0.0)
        belief.update([], self.known, self.mesh, 0.25)
        belief.update([], self.known, self.mesh, 0.5)
        self.assertNotIn(7, belief.tracks)

        reborn = detection(px=-0.8, py=0.7, vx=-0.2, vy=0.4)
        belief.update([reborn], self.unknown, self.mesh, 0.75)
        track = belief.tracks[7]
        np.testing.assert_allclose(track.mean, [-0.8, 0.7, -0.2, 0.4])
        self.assertEqual(track.missed_steps, 0)
        self.assertTrue(track.visible)
        self.assertAlmostEqual(track.existence, 0.999)

    def test_reobservation_reduces_covariance(self):
        belief = BayesianRFSBelief()
        belief.update([detection()], self.unknown, self.mesh, 0.0)
        belief.update([], self.unknown, self.mesh, 0.25)
        before = np.trace(belief.tracks[7].covariance)
        belief.update([detection(px=0.25)], self.unknown, self.mesh, 0.5)
        after = np.trace(belief.tracks[7].covariance)
        self.assertLess(after, before)
        self.assertAlmostEqual(belief.tracks[7].existence, 0.999)

    def test_exact_coordinate_reobservation_does_not_lag(self):
        belief = BayesianRFSBelief()
        belief.update([detection()], self.unknown, self.mesh, 0.0)
        belief.update([], self.unknown, self.mesh, 0.25)
        belief.update(
            [detection(px=-0.4, py=0.6, vx=-0.7, vy=0.2)],
            self.unknown,
            self.mesh,
            0.5,
        )
        np.testing.assert_allclose(
            belief.tracks[7].mean,
            [-0.4, 0.6, -0.7, 0.2],
            atol=2e-3,
        )

    def test_bayesian_buffer_expands_but_fixed_does_not(self):
        bayes = BayesianRFSBelief(mode="bayes")
        fixed = BayesianRFSBelief(mode="fixed")
        for belief in (bayes, fixed):
            belief.update([detection()], self.unknown, self.mesh, 0.0)
            belief.update([], self.unknown, self.mesh, 0.25)
        bayes_out = bayes.output(self.unknown, self.mesh, 8, 0.3, 0.16)
        fixed_out = fixed.output(self.unknown, self.mesh, 8, 0.3, 0.16)
        self.assertGreater(bayes_out.uncertainty_buffer[0, -1],
                           bayes_out.uncertainty_buffer[0, 0])
        np.testing.assert_allclose(
            fixed_out.uncertainty_buffer,
            RFSConfig().fixed_uncertainty_radius,
        )

    def test_discounted_gamma_density_is_finite(self):
        belief = BayesianRFSBelief()
        for step in range(20):
            observations = [detection(i, px=-1.0 + 0.2 * i) for i in range(5)]
            belief.update(observations, self.unknown, self.mesh, step * 0.25)
        output = belief.output(self.unknown, self.mesh, 4, 0.3, 0.16)
        self.assertGreater(output.density_mean, 0.0)
        self.assertTrue(np.isfinite(output.density_std))
        self.assertTrue(np.all((output.occupancy_probability >= 0.0)
                               & (output.occupancy_probability <= 1.0)))

    def test_primary_filter_does_not_invent_unobserved_tracks(self):
        belief = BayesianRFSBelief(mode="bayes")
        belief.update([], self.unknown, self.mesh, 0.0)
        output = belief.output(self.unknown, self.mesh, 4, 0.3, 0.16)
        self.assertEqual(output.track_count, 0)
        self.assertEqual(output.entities.shape, (0, 5))
        self.assertTrue(np.all(output.occupancy_probability == 0.0))


class MatchedEvaluationTest(unittest.TestCase):
    def test_actual_swept_collision_between_clear_endpoints(self):
        from continuous_mpc_gate import swept_min_clearance
        r0, r1 = np.array([-1.,0.]), np.array([1.,0.])
        value = swept_min_clearance(r0,r1,np.array([[0.,-1.]]),np.array([[0.,1.]]),np.array([.6]))
        self.assertAlmostEqual(value,-.6)
        value = swept_min_clearance(r0,r1,np.array([[0.,2.]]),np.array([[0.,2.]]),np.array([.6]))
        self.assertAlmostEqual(value,1.4)

    def test_noise_is_keyed_by_identity_and_time(self):
        from continuous_mpc_gate import ObservationAdapter
        adapter = ObservationAdapter("bayes",position_noise_std=.1,velocity_noise_std=.2,
                                     detection_probability=.8,observation_seed=123)
        a,b = detection(3),detection(7)
        together = {e["id"]:e for e in adapter._corrupt_detections([a,b],.5)}
        reverse = {e["id"]:e for e in adapter._corrupt_detections([b,a],.5)}
        self.assertEqual(together,reverse)
        alone = adapter._corrupt_detections([b],.5)
        self.assertEqual(alone,[together[7]] if 7 in together else [])

    def test_covariance_intervention_preserves_mean_existence(self):
        from types import SimpleNamespace
        from evaluate_matched_safety import MatchedAdapter
        from continuous_mpc_gate import ObservationAdapter
        grid = np.full((20,20),.5)
        sensor = SimpleNamespace(sensor_grid=grid,_mesh=mesh(),visible_ids=[7],res=.25)
        robot = SimpleNamespace(px=0.,py=-2.,vx=0.,vy=0.,radius=.3,gx=0.,gy=4.)
        state = SimpleNamespace(policy_entities=[detection()],self_state=robot,provenance="sensor")
        env = SimpleNamespace(occlusion=sensor,occlusion_enabled=lambda:True,
                              get_policy_state=lambda:state,global_time=0.)
        calibration = {"variance_by_visibility_horizon":np.full((2,16),.1).tolist(),
                       "static_scale":1.,"ewma_alpha":.1}
        full = ObservationAdapter("bayes").read(env)
        static = MatchedAdapter("bayes",method="static_cov",calibration=calibration).read(env)
        np.testing.assert_array_equal(full.entities,static.entities)
        np.testing.assert_array_equal(full.human_segment_end,static.human_segment_end)
        np.testing.assert_array_equal(full.human_existence,static.human_existence)
        self.assertFalse(np.array_equal(full.human_position_covariance,static.human_position_covariance))

    def test_old_search_reuses_current_risk_math(self):
        from evaluate_matched_safety import planner_class
        cls = planner_class("old")
        self.assertIs(cls._belief_collision_hazard,ContinuousCEMMPC._belief_collision_hazard)
        self.assertIs(cls._cost,ContinuousCEMMPC._cost)
        self.assertIsNot(cls.plan,ContinuousCEMMPC.plan)

    def test_report_factorial_and_matched_selection(self):
        from dataclasses import asdict
        import tempfile
        from pathlib import Path
        from continuous_mpc_gate import EpisodeResult
        from evaluate_matched_safety import save,make_report,summarize,SCENES
        import json
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for method,search in (("bayes","new"),("static_cov","new"),("bayes","old"),("static_cov","old")):
                for scene in SCENES:
                    rows = [asdict(EpisodeResult("bayes",scene[2],scene[1],case,"reach_goal",1,0,0,
                        10.,10.,8.,.2,1.,1.,40,"hash",actual_min_clearance=.2,
                        success_without_overlap=1,plan_step_ms=[1.],pipeline_step_ms=[2.])) for case in range(3)]
                    save(out/f"formal_{method}_{search}_{scene[0]}.json",{
                        "protocol":{"method":method,"point":2,"search":search,"scene":scene},
                        "episodes":rows,"summary":summarize(rows)})
            save(out/"frozen_selection.json",{"points":dict.fromkeys(("bayes","static_cov","ewma","conformal","fixed"),2)})
            make_report(out)
            result = json.loads((out/"analysis.json").read_text())
            self.assertEqual(result["search_covariance_interaction"]["success"]["delta"],0.)
            self.assertFalse(result["matched_dev_selected"]["static_cov"]["supports_predeclared_safety_and_time"])


class MPCTerminalSemanticsTest(unittest.TestCase):
    @staticmethod
    def observation():
        return PlannerObservation(
            robot_xy=np.array([0.0, 0.0]),
            robot_velocity=np.zeros(2),
            robot_radius=0.3,
            goal_xy=np.array([0.25, 0.0]),
            entities=np.empty((0, 5)),
            human_segment_start=None,
            human_segment_end=None,
            human_uncertainty_buffer=None,
            human_position_covariance=None,
            human_existence=None,
            human_visible=None,
            unknown=None,
            occupancy_probability=None,
            provenance="unit",
        )

    def test_steps_after_goal_are_inactive(self):
        obs = self.observation()
        positions = np.array(
            [[[0.1, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]
        )
        active, reached, _ = ContinuousCEMMPC._active_until_goal(positions, obs)
        np.testing.assert_array_equal(active, [[True, False, False, False]])
        np.testing.assert_array_equal(reached, [[True, False, False, False]])

    def test_post_goal_risk_does_not_make_trajectory_infeasible(self):
        config = MPCConfig(horizon=4, near_chance_limit=0.1, chance_limit=0.1)
        planner = ContinuousCEMMPC(config)
        obs = self.observation()
        controls = np.array([[[0.4, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        positions = obs.robot_xy[None, None, :] + np.cumsum(
            controls * config.dt, axis=1
        )
        risk = np.array([[0.0, 20.0, 20.0, 20.0]])
        full, first = planner._combined_clearance(
            controls, obs, positions, None, None, risk
        )
        self.assertGreaterEqual(full[0], 0.0)
        self.assertGreaterEqual(first[0], 0.0)


class RouteMPCTest(unittest.TestCase):
    def setUp(self):
        self.cfg = MPCConfig(population=64, iterations=2)
        self.planner = ContinuousCEMMPC(self.cfg)
        self.obs = MPCTerminalSemanticsTest.observation()
        self.obs.goal_xy = np.array([0.0, 8.0])

    def test_bends_return_toward_goal_without_averaging(self):
        routes = self.planner._route_seeds(self.obs)
        self.assertLess(routes[0, 0, 0], 0)
        self.assertGreater(routes[1, 0, 0], 0)
        self.assertGreater(routes[0, -1, 0], 0)
        self.assertLess(routes[1, -1, 0], 0)
        self.assertTrue(np.all(routes[:, -1, 1] > 0))
        np.testing.assert_array_equal(routes[2, :4], 0.0)

    def test_routes_respect_actuator_limits(self):
        self.obs.robot_velocity = np.array([0.6, 0.2])
        routes = self.planner._route_seeds(self.obs)
        previous = np.concatenate((np.broadcast_to(self.obs.robot_velocity,
                                                   (3, 1, 2)), routes[:, :-1]), axis=1)
        self.assertLessEqual(np.linalg.norm(routes, axis=2).max(), self.cfg.v_max + 1e-12)
        self.assertLessEqual(np.linalg.norm(routes - previous, axis=2).max(),
                             self.cfg.a_max * self.cfg.dt + 1e-12)

    def test_elites_prefer_feasible_but_do_not_reward_extra_clearance(self):
        indices = self.planner._elite_indices(
            np.array([1., 4., 3., 0.]), np.array([-.1, 10., .1, -.2]),
            np.array([.1, .1, .1, -.1]), 4)
        np.testing.assert_array_equal(indices, [2, 1, 0, 3])

    def test_total_search_budget_unchanged_and_repeatable(self):
        with patch.object(self.planner, "_cost", wraps=self.planner._cost) as cost:
            action, _ = self.planner.plan(self.obs, 17)
        self.assertEqual(cost.call_count, self.cfg.iterations)
        self.assertEqual([call.args[0].shape[0] for call in cost.call_args_list],
                         [self.cfg.population] * self.cfg.iterations)
        controls = self.planner.last_controls.copy()
        np.testing.assert_array_equal(action, controls[0])
        self.planner.reset()
        second, _ = self.planner.plan(self.obs, 17)
        np.testing.assert_array_equal(action, second)
        np.testing.assert_array_equal(controls, self.planner.last_controls)

    def test_empty_crowd_makes_progress(self):
        action, _ = self.planner.plan(self.obs, 4)
        self.assertGreater(action[1], 0.0)
        self.assertTrue(np.isfinite(action).all())

    def test_bounded_probability_matches_full_integral(self):
        horizon = self.cfg.horizon
        self.obs.entities = np.array([[0., 0., 0., 0., .3],
                                     [20., 0., 0., 0., .3]])
        self.obs.human_segment_end = np.repeat(self.obs.entities[:, None, :2], horizon, axis=1)
        variances = np.geomspace(1e-5, 3., horizon)
        self.obs.human_position_covariance = np.broadcast_to(
            variances[None, :, None, None] * np.eye(2), (2, horizon, 2, 2))
        self.obs.human_existence = np.array([.7, .9])
        positions = np.zeros((100, horizon, 2))
        positions[:, :, 0] = np.linspace(0., 25., 100)[:, None]
        controls = np.zeros_like(positions)
        radius = self.obs.robot_radius + .3 + self.cfg.human_margin
        distance2 = np.square(positions[:, None] - self.obs.human_segment_end[None]).sum(axis=3)
        exact = ncx2.cdf(radius ** 2 / variances, 2., distance2 / variances)
        exact = -np.log1p(-np.clip(exact * self.obs.human_existence[None, :, None],
                                 0., 1. - 1e-12)).sum(axis=1)
        bounded = self.planner._belief_collision_hazard(controls, self.obs, positions)
        np.testing.assert_allclose(bounded, exact, atol=2e-12, rtol=1e-12)
        self.assertTrue(np.all(bounded >= exact - 1e-13))

    def test_small_covariance_integral_against_cartesian_quadrature(self):
        radius, sigma = .76, np.sqrt(1e-5)
        for offset in (-2., 0., 2., 8., 6000.):
            distance = radius + offset * sigma
            lower = max(-12., (-radius - distance) / sigma)
            upper = min(12., (radius - distance) / sigma)
            def integrand(z):
                extent = np.sqrt(max(0., radius ** 2 - (distance + sigma * z) ** 2))
                return np.exp(-.5 * z * z) / np.sqrt(2. * np.pi) * (2. * ndtr(extent / sigma) - 1.)
            expected = quad(integrand, lower, upper, epsabs=1e-10)[0] if upper > lower else 0.
            actual = ncx2.cdf((radius / sigma) ** 2, 2., (distance / sigma) ** 2)
            self.assertAlmostEqual(actual, expected, delta=1e-8)

    def test_zero_variance_probability_is_deterministic(self):
        self.obs.entities = np.array([[0., 0., 0., 0., .3]])
        self.obs.human_segment_end = np.zeros((1, self.cfg.horizon, 2))
        self.obs.human_position_covariance = np.zeros((1, self.cfg.horizon, 2, 2))
        self.obs.human_existence = np.array([.8])
        positions = np.full((2, self.cfg.horizon, 2), 10.)
        positions[0] = 0.
        hazard = self.planner._belief_collision_hazard(np.zeros_like(positions), self.obs, positions)
        np.testing.assert_allclose(-np.expm1(-hazard[0]), .8)
        np.testing.assert_array_equal(hazard[1], 0.)


class ObservationFrameTests(unittest.TestCase):
    def test_full_observation_has_no_unknown_cells_and_preserves_all_people(self):
        from types import SimpleNamespace as NS
        from continuous_mpc_gate import policy_observation_frame, ObservationAdapter
        humans = [NS(px=float(i),py=0.,vx=0.,vy=.2,radius=.3) for i in range(20)]
        robot = NS(px=0.,py=-2.,vx=0.,vy=0.,radius=.3,gx=0.,gy=4.)
        state = NS(human_states=humans,self_state=robot)
        env = NS(occlusion=None,occlusion_enabled=lambda:False,
                 get_policy_state=lambda:state,global_time=0.,human_num=20)
        _, entities, visible, grid, _, _ = policy_observation_frame(env)
        self.assertEqual(len(entities),20)
        self.assertEqual(len(visible),20)
        self.assertFalse(np.any(grid == .5))
        observation = ObservationAdapter('bayes').read(env)
        self.assertEqual(len(observation.entities),20)

    def test_cv_memory_never_creates_an_unobserved_identifier(self):
        from types import SimpleNamespace as NS
        from evaluate_matched_safety import CVMemoryAdapter
        sensor = NS(sensor_grid=np.full((40,40),.5),_mesh=mesh(),visible_ids=[7],res=.25)
        robot = NS(px=0.,py=-2.,vx=0.,vy=0.,radius=.3,gx=0.,gy=4.)
        state = NS(policy_entities=[detection()],self_state=robot,provenance='sensor')
        env = NS(occlusion=sensor,occlusion_enabled=lambda:True,
                 get_policy_state=lambda:state,global_time=0.)
        adapter = CVMemoryAdapter('sensor',max_age=2.)
        adapter.read(env)
        state.policy_entities=[]
        sensor.visible_ids=[]
        env.global_time=1.
        obs=adapter.read(env)
        self.assertEqual(adapter.reported_ids,[7])
        self.assertIsNone(obs.human_position_covariance)
        env.global_time=2.25
        self.assertEqual(len(adapter.read(env).entities),0)


class CalibrationCausalityTests(unittest.TestCase):
    def fixture(self):
        from dataclasses import replace
        from types import SimpleNamespace as NS
        from evaluate_belief_quality import PairedCalibrationAudit
        from evaluate_matched_safety import config
        cfg=replace(config(),horizon=2)
        calibration={'audit_conformal_quantile_radii':np.ones((2,2,4)).tolist()}
        audit=PairedCalibrationAudit(cfg,calibration,(0.,0.,1.),1)
        obs=NS(human_segment_end=np.zeros((1,2,2)),
               human_position_covariance=np.tile(np.eye(2),(1,2,1,1)))
        audit.shadows={name:NS(reported_ids=[0],read=lambda env:obs) for name in audit.METHODS[1:]}
        adapter=NS(reported_ids=[0],rfs=NS(tracks={0:NS(visible=True,missed_steps=0)}))
        return audit,obs,adapter

    def test_forecasts_are_not_scored_before_the_future_arrives(self):
        from types import SimpleNamespace as NS
        audit,obs,adapter=self.fixture()
        audit(NS(),obs,adapter,0)
        self.assertEqual(audit.scored,0)
        audit(NS(humans=[NS(get_position=lambda:np.array([1.,0.]))]),obs,adapter,1)
        result=audit.result()
        self.assertEqual(result['scored_forecasts'],1)
        self.assertEqual(result['terminal_censored_forecasts'],3)
        self.assertEqual(len(result['strata']),4)
        for row in result['strata']:
            self.assertEqual(row['coverage_hits'],[1,1,1,1])
        self.assertIsNone([r for r in result['strata'] if r['method']=='conformal'][0]['conditional_nll_sum'])

    def test_duplicate_observation_is_rejected(self):
        from types import SimpleNamespace as NS
        audit,obs,adapter=self.fixture()
        audit(NS(),obs,adapter,0)
        with self.assertRaises(RuntimeError):audit(NS(),obs,adapter,0)


class ReobservationTests(unittest.TestCase):
    def test_joint_cloud_preserves_predictive_moments(self):
        from reobservation_mpc import predictive_cloud
        model=BayesianRFSBelief(RFSConfig())
        transition,process=model._transition()
        mu=np.array([1.,2.,.1,-.3]);cov=np.diag([.1,.2,.3,.4])
        cloud=predictive_cloud(mu,cov,16,.25,.55,4)
        for h in range(17):
            np.testing.assert_allclose(cloud[:,h].mean(axis=0),mu,atol=1e-12)
            np.testing.assert_allclose(np.cov(cloud[:,h],rowvar=False,bias=True),cov,atol=1e-12)
            mu=transition@mu;cov=transition@cov@transition.T+process

    def test_visibility_depends_on_robot_position(self):
        from reobservation_mpc import detection_likelihood
        likelihood=detection_likelihood(np.array([[0.,0.],[0.,2.]]),np.array([[2.,0.]]),.3,
            np.array([[1.,0.]]),np.array([.4]),5.,.8)
        np.testing.assert_array_equal(likelihood[:,0],[0.,.8])
        far=detection_likelihood(np.array([[0.,0.]]),np.array([[10.,0.]]),.3,
            np.empty((0,2)),np.empty(0),5.,1.)
        self.assertEqual(far[0,0],0.)

    def test_posterior_probability_and_total_moments(self):
        from reobservation_mpc import condition_cloud,predictive_cloud
        cloud=predictive_cloud(np.zeros(4),np.eye(4),4,.25,.55,8)
        likelihood=(cloud[:,2,0]>0.).astype(float)[None]
        probability,existence,means,covs=condition_cloud(cloud,likelihood,.7)
        self.assertAlmostEqual(float(probability.sum()),1.)
        self.assertAlmostEqual(float((probability*existence).sum()),.7)
        weights=probability*existence/.7
        expected=np.einsum('tb,tbhd->hd',weights,means)
        np.testing.assert_allclose(expected,cloud[:,:,:2].mean(axis=0),atol=1e-12)
        second=np.einsum('tb,tbhde->hde',weights,covs+means[..., :,None]*means[...,None,:])
        empirical=np.einsum('whd,whe->hde',cloud[:,:,:2],cloud[:,:,:2])/len(cloud)
        np.testing.assert_allclose(second,empirical,atol=1e-12)

    def fixture(self, missed=True):
        from reobservation_mpc import ReobservationObservation
        from continuous_mpc_gate import MPCConfig
        cfg=MPCConfig(population=64,iterations=2,horizon=4)
        xy=np.array([[1.,0.,0.,0.,.3]])
        segments=np.tile(xy[:,None,:2],(1,4,1))
        transition,process=BayesianRFSBelief(RFSConfig())._transition()
        covariance=.1*np.eye(4);future_cov=[]
        for _ in range(4):
            covariance=transition@covariance@transition.T+process
            future_cov.append(covariance[:2,:2].copy())
        obs=ReobservationObservation(robot_xy=np.array([0.,-1.]),robot_velocity=np.zeros(2),
            robot_radius=.3,goal_xy=np.array([0.,3.]),entities=xy,
            human_segment_start=segments.copy(),human_segment_end=segments.copy(),
            human_uncertainty_buffer=None,human_position_covariance=np.array(future_cov)[None],
            human_existence=np.array([.8]), human_visible=None,unknown=None,occupancy_probability=None,provenance='synthetic',
            state_covariances=np.tile(.1*np.eye(4),(1,1,1)),missed=np.array([missed]))
        return cfg,obs

    def test_shared_prefix_budget_and_repeatability(self):
        from reobservation_mpc import ReobservationCEMMPC,BlindContingencyCEMMPC
        cfg,obs=self.fixture()
        for cls in (ReobservationCEMMPC,BlindContingencyCEMMPC):
            planner=cls(cfg)
            with patch.object(planner,'_cost',wraps=planner._cost) as cost:
                action,_=planner.plan(obs,15)
            self.assertEqual(sum(c.args[0].shape[0] for c in cost.call_args_list),cfg.population*cfg.iterations)
            np.testing.assert_array_equal(planner.last_tree[0,:2],planner.last_tree[1,:2])
            self.assertLessEqual(np.linalg.norm(action),cfg.v_max+1e-12)
            tree=planner.last_tree.copy();planner.reset()
            np.testing.assert_array_equal(action,planner.plan(obs,15)[0])
            np.testing.assert_array_equal(tree,planner.last_tree)

    def test_no_missing_target_is_exact_legacy_path(self):
        from reobservation_mpc import ReobservationCEMMPC
        from continuous_mpc_gate import ContinuousCEMMPC
        cfg,obs=self.fixture(False)
        a=ContinuousCEMMPC(cfg);b=ReobservationCEMMPC(cfg)
        np.testing.assert_array_equal(a.plan(obs,15)[0],b.plan(obs,15)[0])
        np.testing.assert_array_equal(a.last_controls,b.last_controls)

    def test_blind_condition_preserves_passive_trajectory_cost(self):
        from reobservation_mpc import BlindContingencyCEMMPC,predictive_cloud
        cfg,obs=self.fixture()
        planner=BlindContingencyCEMMPC(cfg)
        rng=np.random.default_rng(77)
        controls=planner._project_controls(rng.normal(size=(8,cfg.horizon,2)),obs.robot_velocity)
        trees=np.repeat(controls[:,None],2,axis=1)
        cloud=predictive_cloud(obs.entities[0,:4],obs.state_covariances[0],4,.25,.55,88)
        costs,_,_,_,_=planner._branch_metrics(trees,obs,0,cloud)
        positions=obs.robot_xy+np.cumsum(controls*cfg.dt,axis=1)
        expected=planner._cost(controls,obs,positions,planner._human_clearance(controls,obs),
                               None,planner._belief_collision_hazard(controls,obs,positions))
        np.testing.assert_allclose(costs,expected,rtol=1e-10,atol=1e-8)


if __name__ == "__main__":
    unittest.main()
