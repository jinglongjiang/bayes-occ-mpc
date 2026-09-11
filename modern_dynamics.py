"""Common accelerating-unicycle execution for native MPC comparisons."""
from __future__ import annotations

import math

import numpy as np

from continuous_mpc_gate import swept_min_clearance
from unicycle_mpc_gate import UnicycleCEMMPC


_nodes, _weights = np.polynomial.legendre.leggauss(4)
_nodes = (_nodes + 1.) / 2.
_weights = _weights / 2.


def ramp_displacement(heading, speed_start, speed_end, turn, dt):
    """Integrate a constant acceleration and turn rate over one interval.

    Four-node Gaussian quadrature agrees with the analytic integral to below
    1e-10 metres throughout the registered actuator box, including zero turn.
    """
    heading, speed_start, speed_end, turn = np.broadcast_arrays(
        heading, speed_start, speed_end, turn)
    angle = heading[..., None] + turn[..., None] * _nodes
    speed = speed_start[..., None] + (speed_end - speed_start)[..., None] * _nodes
    return np.asarray(dt)[..., None] * np.stack(((speed * np.cos(angle)) @ _weights,
                          (speed * np.sin(angle)) @ _weights), axis=-1)


class ContinuousUnicycleMPC(UnicycleCEMMPC):
    def _rollout(self, samples, obs):
        params = self._project_controls(samples, obs.robot_velocity)
        headings = np.cumsum(params[:, :, 1], axis=1) - params[:, :, 1] + self._heading(obs)
        speeds = np.concatenate((np.full((len(params), 1), self._speed(obs)),
                                 params[:, :-1, 0]), axis=1)
        delta = ramp_displacement(headings, speeds, params[:, :, 0], params[:, :, 1], self.cfg.dt)
        positions = obs.robot_xy[None, None, :] + np.cumsum(delta, axis=1)
        return params, delta / self.cfg.dt, positions


class ContinuousExecutor:
    """Run the same physical motion and collision audit for every planner.

    Humans still make one original ORCA decision per control period. Only the
    robot's motion changes. The old simulator's old-human-velocity collision
    estimate is replaced with swept clearance against the actual human motion.
    """

    def __init__(self, config):
        self.cfg = config

    def __call__(self, env, command):
        from crowd_sim.envs.utils.action import ActionXY

        if env.robot.visible:
            raise ValueError("the registered comparison requires robot.visible=false")
        robot = env.robot
        dt = float(env.time_step)
        start = np.array(robot.get_position(), dtype=float)
        human_start = np.array([h.get_position() for h in env.humans], dtype=float).reshape(-1, 2)
        radii = np.array([h.radius + robot.radius for h in env.humans])
        heading = float(robot.theta)
        speed_start = float(np.hypot(robot.vx, robot.vy))
        speed_end = float(np.clip(command.v, max(0., speed_start - self.cfg.a_max * dt),
                                   min(self.cfg.v_max, speed_start + self.cfg.a_max * dt)))
        turn = float(np.clip(command.r, -self.cfg.omega_max * dt, self.cfg.omega_max * dt))
        delta = ramp_displacement(heading, speed_start, speed_end, turn, dt)
        old_state = robot.get_full_state()
        old_kinematics = robot.kinematics
        robot.theta = (heading + turn) % (2 * math.pi)
        robot.kinematics = "holonomic"
        try:
            ob, reward, _, _, info = env.step(ActionXY(*(delta / dt)))
        finally:
            robot.kinematics = old_kinematics
        if env.states:
            env.states[-1][0] = old_state
        robot.vx = speed_end * math.cos(robot.theta)
        robot.vy = speed_end * math.sin(robot.theta)
        human_end = np.array([h.get_position() for h in env.humans], dtype=float).reshape(-1, 2)
        times = np.linspace(0., 1., 33)
        curve = start + ramp_displacement(heading, speed_start,
            speed_start + times * (speed_end - speed_start), times * turn, times * dt)
        clearance = math.inf
        for i in range(len(times) - 1):
            hs = human_start + times[i] * (human_end - human_start)
            he = human_start + times[i + 1] * (human_end - human_start)
            clearance = min(clearance, swept_min_clearance(curve[i], curve[i + 1], hs, he, radii))
        # A rigorous chord-deviation bound keeps curved-path contacts from being
        # missed. At these limits it is below 0.000017 m, not a safety margin.
        acceleration = abs(speed_end - speed_start) / dt
        curvature_bound = math.hypot(acceleration, max(speed_start, speed_end) * abs(turn) / dt)
        clearance -= curvature_bound * (dt / 32.) ** 2 / 8.
        reached = np.linalg.norm(np.array(robot.get_position()) - robot.get_goal_position()) < robot.radius
        timed_out = float(env.global_time) >= env.time_limit - 1e-6
        collided = clearance < -1e-9
        event = "collision" if collided else "reach_goal" if reached else "timeout" if timed_out else "nothing"
        info = dict(info, event=event, dmin=clearance, physical_clearance=clearance,
                    acceleration_clip=abs(speed_end - float(command.v)),
                    executed_speed=speed_end, executed_turn=turn,
                    execution_model="continuous_unicycle")
        return ob, reward, event in ("collision", "reach_goal"), timed_out, info
