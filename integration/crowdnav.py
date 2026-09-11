"""Bayes-only legal observation adapter; no oracle or alternative-method dispatch."""
import numpy as np
from scipy.special import ndtr
from nav.belief import BayesianBelief, BeliefConfig
from nav.contracts import PlannerObservation

def policy_observation_frame(env):
    """One permitted observation interface for full and occluded evaluation."""
    state = env.get_policy_state()
    if env.occlusion_enabled():
        return (state, list(state.policy_entities or []), set(env.occlusion.visible_ids),
                env.occlusion.sensor_grid, env.occlusion._mesh, env.occlusion.res)
    entities = [dict(id=i, px=h.px, py=h.py, vx=h.vx, vy=h.vy, radius=h.radius)
                for i, h in enumerate(state.human_states)]
    state.policy_entities = entities
    state.visible_ids = list(range(len(entities)))
    state.provenance = "full_observation"
    resolution, extent = .25, 5.
    axis = resolution * (np.arange(int(2 * extent / resolution)) + .5) - extent
    mesh = np.meshgrid(axis + state.self_state.px, axis + state.self_state.py)
    sensor = np.zeros(mesh[0].shape, dtype=np.float32)
    for entity in entities:
        occupied = ((mesh[0] - entity['px']) ** 2 + (mesh[1] - entity['py']) ** 2
                    <= entity['radius'] ** 2)
        sensor[occupied] = 1.
    return state, entities, set(state.visible_ids), sensor, mesh, resolution


class BayesObservationAdapter:
    def __init__(self, config, belief_config=None, detection_probability=1., observation_seed=0):
        self.cfg = config
        self.belief_config = belief_config or BeliefConfig(
            dt=config.dt, acceleration_std=config.acceleration_std)
        if self.belief_config.dt != config.dt:
            raise ValueError("belief and planner time steps differ")
        if not 0 <= detection_probability <= 1:
            raise ValueError("invalid sensor detection probability")
        self.detection_probability = detection_probability
        self.observation_seed = observation_seed
        self.belief = BayesianBelief(self.belief_config)
        self.reported_ids = []
        self.detected_entities = []

    def reset(self):
        self.belief.reset()
        self.reported_ids = []
        self.detected_entities = []

    def _corrupt_detections(self, entities, timestamp):
        cfg = self.belief_config
        if not (cfg.position_measurement_std > 0 or cfg.velocity_measurement_std > 0
                or self.detection_probability < 1):
            return entities
        observed = []
        step = int(round(timestamp/self.cfg.dt))
        for source in entities:
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.observation_seed, step, int(source["id"]) & 0xffffffff]))
            draw = rng.standard_normal(5)
            if float(ndtr(draw[0])) > self.detection_probability:
                continue
            entity = dict(source)
            entity["px"] += float(draw[1]*cfg.position_measurement_std)
            entity["py"] += float(draw[2]*cfg.position_measurement_std)
            entity["vx"] += float(draw[3]*cfg.velocity_measurement_std)
            entity["vy"] += float(draw[4]*cfg.velocity_measurement_std)
            observed.append(entity)
        return observed

    def read(self, env):
        state, entities, visible_ids, sensor, mesh, _ = policy_observation_frame(env)
        if any(e["id"] not in visible_ids for e in entities):
            raise RuntimeError("hidden-state leakage in Bayes observation")
        entities = self._corrupt_detections(entities, env.global_time)
        self.detected_entities = entities
        self.belief.update(entities, sensor, mesh, env.global_time)
        output = self.belief.output(self.cfg.horizon)
        self.reported_ids = [t.identifier for t in self.belief.tracks.values()
                             if t.visible or t.existence >= self.belief.cfg.report_existence]
        if len(self.reported_ids) != len(output.entities):
            raise RuntimeError("track/output ordering mismatch")
        robot = state.self_state
        return PlannerObservation(
            robot_xy=np.array([robot.px,robot.py],dtype=np.float64),
            robot_velocity=np.array([robot.vx,robot.vy],dtype=np.float64),
            robot_radius=float(robot.radius),
            goal_xy=np.array([robot.gx,robot.gy],dtype=np.float64),
            entities=output.entities,
            human_segment_start=output.segment_start,
            human_segment_end=output.segment_end,
            human_uncertainty_buffer=None,
            human_position_covariance=output.position_covariance,
            human_existence=output.existence,
            human_visible=output.visible,
            unknown=None,
            occupancy_probability=None,
            provenance=state.provenance,
            robot_heading=float(getattr(robot,"theta",0.0)))
