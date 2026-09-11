"""Isolated development checks for the modern-planner adapters.

The archived formal experiment is never modified by this runner. Native MPC
modules remain intact; reference/weight choices are tested on synthetic goal
recovery cases before any crowd benchmark is consulted.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import time
import hashlib
import xmlrpc.client
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from modern_worker import BridgeController
from continuous_mpc_gate import ObservationAdapter


ROOT = Path("/home/abc/temp/modern/repair")
BUILD = Path("/home/abc/workspace/bayes_occ_mpc/build_modern")


class JointObservationAdapter(ObservationAdapter):
    """Expose the existing filter state, without adding an observation source."""

    def read(self, env):
        obs = super().read(env)
        if self.rfs is None:
            raise ValueError("joint predictions require the shared legal-history filter")
        obs.human_state_covariance = np.asarray([
            self.rfs.tracks[i].covariance for i in self.reported_ids
        ], dtype=float).reshape(-1, 4, 4)
        obs.human_track_ids = np.asarray(self.reported_ids, dtype=int)
        obs.acceleration_std = self.rfs.cfg.acceleration_std
        if len(obs.human_state_covariance) != len(obs.entities):
            raise ValueError("joint prediction and entity ordering differ")
        return obs


class CertainExistenceAdapter(JointObservationAdapter):
    def read(self, env):
        obs = super().read(env)
        if obs.human_existence is not None:
            obs.human_existence = np.ones_like(obs.human_existence)
        return obs


class JointBridgeController(BridgeController):
    """SH receives a CV state covariance, rather than invented independent increments."""

    sample_seed = 0

    def _exchange(self, line):
        reply = super()._exchange(line)
        if line.startswith("STEP "):
            self.last_native = {}
            for field in reply.split()[7:]:
                key, separator, value = field.partition("=")
                if separator:
                    number = float(value)
                    self.last_native[key] = number if math.isfinite(number) else None
            self.last_request = line
        return reply

    def __init__(self, *args, **kwargs):
        self._unwrapped_heading = None
        guidance_seed = int(kwargs.pop("guidance_seed", 0))
        horizon = args[1] if len(args) > 1 else kwargs["horizon"]
        dt = args[2] if len(args) > 2 else kwargs["dt"]
        self._ros_server = xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"])
        code, message, global_parameters = self._ros_server.getParam("/modern_repair", "/")
        if code != 1:
            raise RuntimeError(f"cannot read native ROS parameters: {message}")
        parameters = {key: copy.deepcopy(global_parameters[key])
                      for key in ("guidance_planner", "scenario_module", "visuals", "target_frame")
                      if key in global_parameters}
        guidance = parameters["guidance_planner"]
        guidance.update(T=horizon * dt, N=horizon, seed=guidance_seed,
                        max_velocity=float(kwargs.get("speed_limit", 1.)), max_acceleration=2.)
        if not isinstance(guidance.get("debug"), dict):
            guidance["debug"] = {"output": False, "visuals": False}
        parameters["clock_frequency"] = 1. / dt
        self._namespace = f"/modern_repair/p{os.getpid()}_{time.time_ns()}"
        code, message, _ = self._ros_server.setParam("/modern_repair", self._namespace, parameters)
        if code != 1:
            raise RuntimeError(f"cannot isolate native parameters: {message}")
        old_namespace = os.environ.get("ROS_NAMESPACE")
        try:
            os.environ["ROS_NAMESPACE"] = self._namespace
            super().__init__(*args, **kwargs)
            capabilities = self._exchange("CAPABILITIES").split()[1:]
            required = {"endpoint_velocity"}
            if self.base_variant == "shmpc":
                required.add("joint_cv")
            if not required.issubset(capabilities):
                raise RuntimeError("bridge does not implement the requested repaired protocol")
        except Exception:
            if hasattr(self, "process"):
                self.close()
            else:
                self._ros_server.deleteParam("/modern_repair", self._namespace)
            raise
        finally:
            if old_namespace is None:
                os.environ.pop("ROS_NAMESPACE", None)
            else:
                os.environ["ROS_NAMESPACE"] = old_namespace

    def close(self):
        try:
            super().close()
        finally:
            try:
                self._ros_server.deleteParam("/modern_repair", self._namespace)
            finally:
                self._ros_server.__exit__(None, None, None)

    def to_request(self, obs):
        heading = float(obs.robot_heading)
        previous = getattr(self, "_unwrapped_heading", None)
        if previous is None:
            continuous_heading = math.atan2(math.sin(heading), math.cos(heading))
        else:
            continuous_heading = previous + math.atan2(math.sin(heading - previous), math.cos(heading - previous))
        self._unwrapped_heading = continuous_heading
        obs = copy.copy(obs)
        obs.robot_heading = continuous_heading
        request = super().to_request(obs)
        if self.base_variant != "shmpc":
            return request
        covariances = getattr(obs, "human_state_covariance", None)
        if covariances is None:
            if len(obs.entities):
                raise ValueError("SH joint sampler requires current position-velocity covariance")
            covariances = np.empty((0, 4, 4))
        covariance = np.asarray(covariances, dtype=float)
        if covariance.shape != (len(obs.entities), 4, 4) or not np.all(np.isfinite(covariance)):
            raise ValueError("invalid current state covariance")
        parts = ["JOINT_CV", str(int(self.sample_seed)),
                 str(float(getattr(obs, "acceleration_std", .55))), str(len(covariance))]
        parts.extend(format(float(v), ".17g") for v in covariance.ravel())
        return request + " " + " ".join(parts)

    def reset(self):
        super().reset()
        self._unwrapped_heading = None

    def _ellipses(self, obs):
        if self.base_variant == "shmpc":
            covariance = obs.human_position_covariance
            return None if covariance is None or not np.size(covariance) else self._to_ellipse(covariance)
        return super()._ellipses(obs)


def runtime_environment():
    os.environ.setdefault("ROS_MASTER_URI", "http://localhost:11311")
    prefixes = {
        "ROS_PACKAGE_PATH": [str(BUILD / "shmpc_n24/src"), str(BUILD / "tmpc/src"),
                             "/opt/ros/noetic/share"],
        "LD_LIBRARY_PATH": ["/opt/ros/noetic/lib"],
    }
    for name, first in prefixes.items():
        entries = first + os.environ.get(name, "").split(os.pathsep)
        os.environ[name] = os.pathsep.join(dict.fromkeys(p for p in entries if p))
    os.environ["MODERN_BRIDGE_LOG_DIR"] = str(ROOT / "bridge_logs")


def write_result(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance(controller, cfg):
    paths = [Path(__file__), Path(__file__).with_name("modern_worker.py"),
             Path(__file__).with_name("modern_dynamics.py"),
             Path(__file__).with_name("continuous_mpc_gate.py"),
             Path(__file__).with_name("unicycle_mpc_gate.py"),
             Path(__file__).with_name("bayesian_rfs.py"),
             Path(__file__).with_name("evaluate_matched_safety.py")]
    from continuous_mpc_gate import DEFAULT_CROWDNAV
    paths += [DEFAULT_CROWDNAV / relative for relative in (
        "crowd_sim/envs/crowd_sim.py", "crowd_sim/envs/utils/state.py",
        "crowd_sim/envs/policy/orca.py", "crowd_nav/configs/env.config")]
    metadata = dict(config=dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg)
    if controller is not None:
        # Record what this process loaded, rather than assuming its workspace
        # name proves that the matching shared libraries were used.
        for line in Path(f"/proc/{controller.process.pid}/maps").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 6 and fields[5].startswith("/"):
                path = Path(fields[5])
                if path.is_file() and ("bayes_occ_mpc" in str(path) or "/opt/ros/" in str(path)):
                    paths.append(path)
        paths += [controller.binary, controller.settings]
        generated = BUILD / controller.variant / "src/mpc_planner/mpc_planner_solver/acados/Solver/Solver.json"
        description = json.loads(generated.read_text())
        metadata["generated_terminal"] = {
            "cost_type": description["cost"]["cost_type_e"],
            "constraint_count": description["dims"]["nh_e"],
            "state_bound_count": description["dims"]["nbx_e"],
        }
        paths.append(generated)
        source = BUILD / controller.variant / "src"
        paths += [p for p in source.rglob("*") if p.is_file() and p.suffix in
                  (".cpp", ".h", ".hpp", ".py", ".yaml") and
                  not any(part in (".git", "__pycache__", "acados", "build") for part in p.parts)]
        # The SH package also reads global ROS parameters. Include them, without
        # changing a parameter server another process may be using.
        with xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"]) as server:
            code, message, ros_parameters = server.getParam("/modern_repair_audit", controller._namespace)
        if code != 1:
            raise RuntimeError(f"cannot archive ROS parameters: {message}")
        metadata["ros_parameters"] = ros_parameters
    metadata["files"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(set(paths))}
    return metadata


def settings(variant, profile, out=ROOT, risk=None):
    source = BUILD / variant / "src/mpc_planner/mpc_planner_jackalsimulator/config/settings.yaml"
    config = yaml.safe_load(source.read_text())
    # Preserve the previously selected development points; do not make the
    # repaired SH adapter artificially stricter while testing correctness.
    config["probabilistic"]["risk"] = (.42 if variant.startswith("shmpc") else .35) if risk is None else risk
    upper = .5 if variant.startswith("shmpc") else 1.
    if not 0. < config["probabilistic"]["risk"] < upper:
        raise ValueError("configured risk is outside the native consumer's domain")
    config["recording"]["enable"] = False
    profiles = {
        "legacy": {"goal": 0.},
        "track": {"contour": .5, "terminal_contouring": 50., "goal": 1.},
        "heading_free": {"contour": .5, "terminal_contouring": 50., "terminal_angle": 0.},
        "terminal": {"contour": 2., "terminal_contouring": 100., "terminal_angle": 0.},
        "balanced": {"contour": 2., "lag": 2., "terminal_contouring": 50., "terminal_angle": 0.},
        "goal_track": {"contour": 2., "terminal_contouring": 50., "terminal_angle": 100., "goal": 10.},
    }
    config["weights"].update(profiles[profile])
    risk_suffix = "" if risk is None else f"_risk{risk:g}"
    target = out / "settings" / f"{variant}_{profile}{risk_suffix}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    return target, config


def empty_observation(position, goal, heading, speed=0.):
    return SimpleNamespace(
        robot_xy=np.array(position, dtype=float), goal_xy=np.array(goal, dtype=float),
        robot_heading=float(heading),
        robot_velocity=float(speed) * np.array([math.cos(heading), math.sin(heading)]),
        entities=np.empty((0, 5)), human_position_covariance=None,
        human_segment_start=None, human_segment_end=None, human_uncertainty_buffer=None,
    )


def goal_probe(variant="tmpc", profiles=None):
    from modern_dynamics import ramp_displacement
    runtime_environment()
    output = ROOT / "goals" / str(time.time_ns()) / f"{variant}.json"
    profiles = profiles or ["legacy", "track", "heading_free", "terminal", "balanced"]
    cases = [
        ("aligned", (0., -4.), math.pi / 2, 0.),
        ("near_lateral", (.6, 3.), math.pi / 2, 1.),
        ("lateral", (1., 0.), math.pi / 2, .5),
        ("away", (0., 0.), -math.pi / 2, 0.),
    ]
    rows = []
    for profile in profiles:
        path, config = settings(variant, profile)
        controller_type = JointBridgeController
        controller = controller_type(variant, config["N"], config["integrator_step"],
                                      settings=str(path), human_margin=.1,
                                      path_overshoot=0.)
        manifest = provenance(controller, config)
        manifest_path = output.parent / f"{variant}_{profile}_provenance.json"
        write_result(manifest_path, manifest)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        try:
            for name, position, heading, speed in cases:
                controller.reset()
                obs = empty_observation((0., -4.), (0., 4.), math.pi / 2)
                controller._reference_path(obs)
                obs = empty_observation(position, (0., 4.), heading, speed)
                trace = []
                for step in range(100):
                    action, elapsed = controller.act(obs)
                    speed_start = float(np.linalg.norm(obs.robot_velocity))
                    velocity = float(np.clip(action[0], max(0., speed_start - .5), min(1., speed_start + .5)))
                    turn = float(np.clip(action[1], -.2, .2))
                    obs.robot_xy += ramp_displacement(obs.robot_heading, speed_start, velocity, turn, .25)
                    obs.robot_heading = (obs.robot_heading + turn) % (2 * math.pi)
                    obs.robot_velocity = velocity * np.array([
                        math.cos(obs.robot_heading), math.sin(obs.robot_heading)])
                    distance = float(np.linalg.norm(obs.robot_xy - obs.goal_xy))
                    trace.append(dict(x=float(obs.robot_xy[0]), y=float(obs.robot_xy[1]),
                                      distance=distance, speed=velocity,
                                      heading=float(obs.robot_heading), turn=turn,
                                      solved=controller.last.success, elapsed_ms=elapsed))
                    if distance < .3:
                        break
                row = dict(profile=profile, case=name, success=distance < .3,
                           nav_time=(step + 1) * .25, min_distance=min(s["distance"] for s in trace),
                           solve_fraction=sum(s["solved"] for s in trace) / len(trace), steps=trace,
                           provenance_file=str(manifest_path), provenance_sha256=manifest_hash,
                           execution_model="continuous_unicycle")
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "steps"}), flush=True)
                write_result(output, rows)
        finally:
            controller.close()
    return rows


def cohort(arms, count, case_start, human_count=5, scenario="circle_crossing", occluded=True,
           profile="track", continuous=False, circle_radius=4., square_width=10., scene_name=None,
           diagnose_candidates=False, native_risk=None, trace=False, bayes_point=4,
           planner_seed_offset=0, config_overrides=None):
    from continuous_mpc_gate import run_episode, DEFAULT_CROWDNAV
    from evaluate_matched_safety import config as point_config
    from unicycle_mpc_gate import UnicycleCEMMPC, unicycle_config
    from modern_dynamics import ContinuousExecutor, ContinuousUnicycleMPC

    runtime_environment()
    output = ROOT / "runs" / str(time.time_ns()) / f"cohort_{scenario}_{human_count}_{case_start}_{count}_{profile}_{'occ' if occluded else 'full'}.json"
    print('OUTPUT', output, flush=True)
    rows = []
    for arm in arms:
        controller = None
        if arm in ("bayes", "bayes_r1"):
            cfg = unicycle_config(point_config(bayes_point))
            cfg = dataclasses.replace(cfg, **(config_overrides or {}))
            planner = (ContinuousUnicycleMPC if continuous else UnicycleCEMMPC)(cfg)
        else:
            variant = arm
            path, native = settings(variant, profile, out=output.parent, risk=native_risk)
            cfg = unicycle_config(point_config(2), horizon=native["N"])
            controller_type = JointBridgeController
            controller = controller_type(variant, cfg.horizon, cfg.dt, settings=str(path),
                                         human_margin=cfg.human_margin, path_overshoot=0.,
                                         guidance_seed=case_start * 97 + planner_seed_offset * 100003)

            class Planner:
                last_diagnostics = {}
                native_steps = []

                def plan(self, obs, seed):
                    controller.sample_seed = seed
                    action, elapsed = controller.act(obs)
                    self.last_diagnostics = dict(solver_success=float(controller.last.success),
                                                 solve_ms=controller.last.solve_ms,
                                                 exit_code=controller.last.exit_code)
                    self.native_steps.append(dict(success=controller.last.success,
                                                  **controller.last_native))
                    if trace:
                        fields = controller._exchange("TRAJECTORY").split()
                        values = np.asarray(fields[2:], dtype=float).reshape(cfg.horizon + 1, 7)
                        if fields[0] != "TRAJECTORY" or int(fields[1]) != cfg.horizon:
                            raise RuntimeError("invalid native trajectory reply")
                        self.native_steps[-1]["trajectory_xy_heading_v_s_a_w"] = values.tolist()
                    if diagnose_candidates and not controller.last.success and self.probes < 3:
                        self.probes += 1
                        candidate = ContinuousUnicycleMPC(unicycle_config(point_config(4), horizon=cfg.horizon))
                        candidate.plan(obs, seed)
                        _, _, positions = candidate._rollout(candidate.last_controls[None], obs)
                        major, minor, tilt = controller._ellipses(obs)
                        radius = obs.robot_radius + obs.entities[:, 4, None] + cfg.human_margin
                        axes = np.stack((major, minor), axis=-1) * math.sqrt(-2 * math.log(.35)) + radius[..., None]
                        delta = positions[0][None] - obs.human_segment_end
                        cosine, sine = np.cos(tilt), np.sin(tilt)
                        # Match the upstream R.T @ D @ R expression exactly.
                        rotated = np.stack((cosine * delta[..., 0] - sine * delta[..., 1],
                                            sine * delta[..., 0] + cosine * delta[..., 1]), axis=-1)
                        constraints = np.sum((rotated / axes) ** 2, axis=-1)
                        rng = np.random.default_rng(seed)
                        knots = rng.uniform([0., -.2], [1., .2], size=(3000, 3, 2))
                        samples = np.stack([np.column_stack([
                            np.interp(np.arange(cfg.horizon), np.linspace(0, cfg.horizon - 1, 3), knot[:, j])
                            for j in range(2)]) for knot in knots])
                        _, _, pool_positions = candidate._rollout(samples, obs)
                        pool_delta = pool_positions[:, None] - obs.human_segment_end[None]
                        rotated_pool = np.stack((cosine * pool_delta[..., 0] - sine * pool_delta[..., 1],
                                                 sine * pool_delta[..., 0] + cosine * pool_delta[..., 1]), axis=-1)
                        min_constraint = np.sum((rotated_pool / axes) ** 2, axis=-1).min(axis=(1, 2))
                        self.native_steps[-1]["candidate_probe"] = dict(
                            cem_diagnostics=candidate.last_diagnostics,
                            tmpc_ellipse_min=float(constraints.min()),
                            tmpc_first_ellipse_min=float(constraints[:, 0].min()),
                            sampled_trajectories=len(samples),
                            sampled_ellipse_feasible=int((min_constraint >= 1.).sum()),
                            sampled_best_min_constraint=float(min_constraint.max()),
                            note="read-only CEM trajectory; ellipse check is not SH scenario feasibility")
                    return action, elapsed

            planner = Planner()
        try:
            manifest = provenance(controller, cfg)
            manifest_path = output.parent / f"{arm}_provenance.json"
            write_result(manifest_path, manifest)
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            for case in range(case_start, case_start + count):
                evidence = []

                def observe_for_audit(env, obs, adapter, step):
                    evidence.append(dict(step=step, robot_xy=obs.robot_xy.tolist(),
                        robot_velocity=obs.robot_velocity.tolist(), heading=obs.robot_heading,
                        entities=obs.entities.tolist(), reported_ids=adapter.reported_ids,
                        visible_ids=[e["id"] for e in adapter.detected_entities],
                        predicted_humans=obs.human_segment_end.tolist(),
                        covariance=obs.human_position_covariance.tolist(),
                        actual_humans_before=[list(h.get_position()) for h in env.humans],
                        actual_velocities_before=[[h.vx, h.vy] for h in env.humans]))

                class AuditExecutor(ContinuousExecutor):
                    def __call__(self, env, command):
                        result = super().__call__(env, command)
                        evidence[-1].update(actual_humans_after=[list(h.get_position()) for h in env.humans],
                            actual_robot_after=list(env.robot.get_position()), physical_clearance=result[-1]["physical_clearance"])
                        return result

                if controller is not None:
                    controller.reset()
                    planner.native_steps = []
                    planner.probes = 0
                else:
                    planner.reset()
                result = run_episode(DEFAULT_CROWDNAV, "bayes", human_count, scenario, case, cfg,
                                     circle_radius=circle_radius, square_width=square_width, time_limit=25,
                                     planner_seed_offset=planner_seed_offset,
                                     planner_type=lambda _: planner,
                                     adapter_factory=CertainExistenceAdapter if arm == "bayes_r1" else JointObservationAdapter,
                                     robot_kinematics="unicycle", occlusion=occluded,
                                     step_observer=observe_for_audit if trace else None,
                                     step_executor=(AuditExecutor(cfg) if trace else ContinuousExecutor(cfg)) if continuous else None)
                row = dataclasses.asdict(result)
                if trace:
                    row["audit_evidence"] = evidence
                if controller is not None:
                    row["native_steps"] = planner.native_steps
                row.update(arm=arm, profile=profile, development_only=True,
                           bayes_point=bayes_point if controller is None else None,
                           planner_seed_offset=planner_seed_offset,
                           native_risk=None if controller is None else native["probabilistic"]["risk"],
                           scene_name=scene_name or scenario, occluded=occluded, robot_visible=False,
                           goal_radius_rule="robot.radius", goal_radius=.3,
                           execution_model="continuous_unicycle" if continuous else "turn_translate",
                           provenance_file=str(manifest_path), provenance_sha256=manifest_hash,
                           runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           binary_sha256=None if controller is None else hashlib.sha256(controller.binary.read_bytes()).hexdigest(),
                           settings_sha256=None if controller is None else hashlib.sha256(controller.settings.read_bytes()).hexdigest())
                rows.append(row)
                # The legacy diagnostics contain NaN for quantities a solver does not expose.
                finite = json.loads(json.dumps(rows), parse_constant=lambda _: None)
                write_result(output, finite)
                print(json.dumps(dict(arm=arm, case=case, event=row['event'],
                                      success=row['success_without_overlap'], collision=row['collision_union'],
                                      nav=row['nav_time'])), flush=True)
        finally:
            if controller is not None:
                controller.close()
    for arm in arms:
        group = [r for r in rows if r['arm'] == arm]
        print('SUMMARY', arm, {k: sum(r[k] for r in group) / len(group)
                              for k in ['success_without_overlap', 'collision_union', 'timeout']}, flush=True)
    return json.loads(json.dumps(rows), parse_constant=lambda _: None)


def suite(arms, count, case_start, full=False, profile="goal_track", native_risks=None):
    from evaluate_matched_safety import SCENES

    output = ROOT / "suites" / str(time.time_ns()) / "results.json"
    print("SUITE_OUTPUT", output, flush=True)
    all_rows = []
    for name, humans, scenario, radius, width in SCENES:
        for arm in arms:
            all_rows += cohort([arm], count, case_start, humans, scenario, not full,
                               profile, True, radius, width, name,
                               native_risk=(native_risks or {}).get(arm))
        finite = json.loads(json.dumps(all_rows), parse_constant=lambda _: None)
        write_result(output, finite)
    expected = len(arms) * len(SCENES) * count
    keys = {(r["arm"], r["scene_name"], r["case_id"]) for r in all_rows}
    if len(all_rows) != expected or len(keys) != expected:
        raise RuntimeError("incomplete or duplicate development suite")
    print("SUITE_DONE", output, "episodes", expected, flush=True)
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["goal", "cohort", "suite"])
    parser.add_argument("--variant", default="tmpc")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--arms", nargs="+", default=["bayes", "tmpc_repair_n8", "shmpc_repair_n8"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--case-start", type=int, default=3650)
    parser.add_argument("--human-count", type=int, default=5)
    parser.add_argument("--scenario", default="circle_crossing")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--profile", default="track")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--diagnose-candidates", action="store_true")
    parser.add_argument("--native-risk", type=float)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.command == 'goal':
        goal_probe(args.variant, args.profiles)
    elif args.command == 'suite':
        suite(args.arms, args.count, args.case_start, args.full, args.profile)
    else:
        cohort(args.arms, args.count, args.case_start, args.human_count,
               args.scenario, not args.full, args.profile, args.continuous,
               diagnose_candidates=args.diagnose_candidates, native_risk=args.native_risk, trace=args.trace)
