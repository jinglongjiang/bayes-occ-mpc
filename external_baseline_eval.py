#!/usr/bin/env python3
"""Matched visible-only ORCA/Mamba baselines for the occlusion benchmark."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from continuous_mpc_gate import (
    DEFAULT_CROWDNAV,
    EpisodeResult,
    _load_modules,
    build_env,
    policy_observation_frame,
    swept_min_clearance,
    summarize,
)


def merge_config(crowdnav_root: Path, env_config) -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    for path in (
        crowdnav_root / "crowd_nav/configs/policy.config",
        crowdnav_root / "crowd_nav/configs/train.config",
    ):
        if not config.read(path):
            raise FileNotFoundError(path)
    for section in env_config.sections():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in env_config.items(section):
            config.set(section, key, value)
    if not config.has_section("sarl"):
        config.add_section("sarl")
    config.set("sarl", "epsilon_start", "0.0")
    config.set("buffer", "seq_len", "24")
    config.set("temporal", "T", "24")
    return config


def load_controller(name: str, crowdnav_root: Path, env_config, checkpoint: Path,
                    device_name="cpu", attngraph_root=None):
    if name == "orca":
        from crowd_sim.envs.policy.orca import ORCA

        if env_config.has_option("orca", "safety_space"):
            env_config.set(
                "orca",
                "safety_space",
                env_config.get("orca", "safety_space").split("#", 1)[0].strip(),
            )
        policy = ORCA()
        policy.configure(env_config)
        policy.set_phase("test")
        return policy, "cpu"

    import torch
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested but unavailable')
    config = merge_config(crowdnav_root, env_config)
    config.set('action_space', 'time_step', '0.25')
    config.set('action_space', 'query_env', 'false')
    if name in ('cadrl', 'sarl', 'lstm'):
        # Original CrowdNav policy settings, not the current Mamba training
        # config. CADRL layer widths are verified against the supplied weights.
        legacy = {
            'rl': {'gamma': '0.9'},
            'om': {'cell_num':'4', 'cell_size':'1', 'om_channel_size':'3'},
            'action_space': {'kinematics':'holonomic','speed_samples':'5',
                             'rotation_samples':'16','sampling':'exponential',
                             'time_step':'0.25','query_env':'false'},
            'cadrl': {'mlp_dims':'64, 32, 1','multiagent_training':'false'},
            'lstm_rl': {'global_state_dim':'50','mlp2_dims':'150, 100, 100, 1',
                        'with_om':'false','with_interaction_module':'false',
                        'multiagent_training':'true'},
            'sarl': {'mlp1_dims':'150, 100','mlp2_dims':'100, 50',
                     'mlp3_dims':'150, 100, 100, 1','attention_dims':'100, 100, 1',
                     'with_om':'false','with_global_state':'true','multiagent_training':'true'},
        }
        config.read_dict(legacy)
        from crowd_nav.policy.cadrl import CADRL
        from crowd_nav.policy.sarl import SARL
        from crowd_nav.policy.lstm_rl import LstmRL
        policy = {'cadrl': CADRL, 'sarl': SARL, 'lstm': LstmRL}[name]()
        policy.configure(config)
    elif name == 'dsrnn':
        from crowd_nav.policy.dsrnn_policy import DSRNNPolicy
        policy = DSRNNPolicy()
        policy.configure(config)
        # This adapter's published network has no missing-human mask. Only its
        # full-observation, variable-cardinality path is admitted in this cohort.
        policy.human_num = env_config.getint('sim', 'human_num')
    elif name == 'attngraph':
        from crowd_nav.policy.attngraph_policy import AttnGraphPolicy
        ag_root = Path(attngraph_root)
        model_dir = checkpoint.parent.parent
        gst_dir = ag_root / 'gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj'
        policy = AttnGraphPolicy(attngraph_root=str(ag_root), model_dir=str(model_dir),
                                 gst_model_dir=str(gst_dir), device=device)
        policy.configure(config)
        policy.sensor_range = math.inf
    elif name == 'mamba':
        from crowd_nav.policy.mamba_rl import warm_start_from_legacy
        from crowd_nav.policy.policy_factory import policy_factory
        policy = policy_factory['mamba'](config, device=device)
    else:
        raise ValueError(name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, (tuple, list)):
        payload = payload[0]
    if hasattr(payload, 'state_dict'):
        payload = payload.state_dict()
    state_dict = payload.get(
        "policy_state",
        payload.get(
            "model_state_dict",
            payload.get("value_state", payload.get("model", payload)),
        ),
    )
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
        }
    if name == 'mamba':
        warm_start_from_legacy(policy, state_dict)
        policy.use_sarl_predict = True
        policy.to(device)
        policy.device = device
    else:
        target = policy if hasattr(policy, 'load_state_dict') else policy.model
        target.load_state_dict(state_dict)
        policy.set_device(device)
        if name == 'dsrnn':
            policy.model.base.human_num = policy.human_num
            policy.model.base.num_edges = policy.human_num + 1
            policy.model.base.spatial_edges = np.arange(1, policy.human_num + 1)
            policy.dsrnn_config.training.cuda = device.type == 'cuda'
    policy.set_phase("test")
    policy.time_step = .25
    if name == 'mamba':
        policy.eval()
    else:
        policy.model.eval()
    reset_controller(policy)
    return policy, str(device)


class VisibleLookahead:
    """CADRL transition facade: never exposes the simulator or hidden people."""

    def __init__(self, policy, state):
        self.policy, self.state = policy, state

    def onestep_lookahead(self, action):
        from crowd_nav.policy.multi_human_rl import MultiHumanRL
        from crowd_sim.envs.utils.action import ActionXY
        robot = self.policy.propagate(self.state.self_state, action)
        humans = [self.policy.propagate(h, ActionXY(h.vx, h.vy))
                  for h in self.state.human_states]
        reward = MultiHumanRL.compute_reward(self.policy, robot, humans)
        return humans, reward, False, None


def predict_batched_value(policy, state, label):
    """Batch the native value queries; keep its actions, reward and discount."""
    import torch
    from crowd_nav.policy.multi_human_rl import MultiHumanRL
    from crowd_sim.envs.utils.action import ActionXY
    if policy.reach_destination(state):
        return ActionXY(0., 0.)
    if policy.with_om:
        raise ValueError('This batched adapter is for the supplied non-OM checkpoints')
    if policy.action_space is None:
        policy.build_action_space(state.self_state.v_pref)
    humans = list(state.human_states)
    if label == 'lstm':
        humans.sort(key=lambda h: np.linalg.norm(np.array(h.position) -
                    np.array(state.self_state.position)), reverse=True)
    next_humans = [policy.propagate(h, ActionXY(h.vx,h.vy)) for h in humans]
    next_robots = [policy.propagate(state.self_state,a) for a in policy.action_space]
    inputs = torch.tensor([[r+h for h in next_humans] for r in next_robots],
                          dtype=torch.float32, device=policy.device)
    rotated = policy.rotate(inputs.reshape(-1,inputs.shape[-1]))
    if label == 'cadrl':
        values = policy.model(rotated).reshape(len(next_robots),len(humans)).min(dim=1).values
    else:
        values = policy.model(rotated.reshape(len(next_robots),len(humans),-1)).reshape(-1)
    rewards = np.array([MultiHumanRL.compute_reward(policy,r,next_humans) for r in next_robots])
    action_values = rewards + policy.gamma ** (policy.time_step * state.self_state.v_pref) * values.detach().cpu().numpy()
    if not np.all(np.isfinite(action_values)):
        raise RuntimeError('nonfinite batched values')
    policy.action_values = action_values.tolist()
    return policy.action_space[int(np.argmax(action_values))]


def reset_controller(policy) -> None:
    if hasattr(policy, "reset_episode_stats"):
        policy.reset_episode_stats()
    if hasattr(policy, "sim"):
        policy.sim = None
    if hasattr(policy, "_last_pref_vel"):
        policy._last_pref_vel = None


def run_episode(env, policy, action_cls, label: str, case_id: int) -> EpisodeResult:
    env.reset(options={"test_case": case_id})
    reset_controller(policy)
    path_length = 0.0
    minimum_clearance = math.inf
    decision_times = []
    hashes = []
    pipeline_times = []
    actual_minimum = math.inf
    overlap_steps = 0
    first_overlap = None
    event = "timeout"
    terminated = truncated = False
    while not (terminated or truncated):
        cycle_start = time.perf_counter()
        state, entities, visible_ids, _, _, _ = policy_observation_frame(env)
        for human, entity in zip(state.human_states, entities):
            human.id = int(entity['id'])
        leaked = [entity["id"] for entity in entities if entity["id"] not in visible_ids]
        if leaked:
            raise RuntimeError(f"hidden-state leakage in {label}: {leaked}")
        payload = np.asarray([
            value
            for entity in entities
            for value in (
                entity["id"], entity["px"], entity["py"],
                entity["vx"], entity["vy"], entity["radius"],
            )
        ], dtype=np.float64)
        hashes.append(hashlib.sha256(payload.tobytes()).hexdigest())
        started = time.perf_counter()
        rng_state = np.random.get_state()
        try:
            if label == 'cadrl':
                policy.env = VisibleLookahead(policy, state)
            if not state.human_states and label in ('cadrl', 'sarl', 'lstm'):
                # Pairwise value networks are undefined on an empty set. Use
                # free-space goal motion, not a fictitious pedestrian token.
                robot = state.self_state
                delta = np.array([robot.gx - robot.px, robot.gy - robot.py])
                speed = min(robot.v_pref, np.linalg.norm(delta) / env.time_step)
                velocity = speed * delta / max(np.linalg.norm(delta), 1e-12)
                action = action_cls(*velocity)
            elif label in ('cadrl','sarl','lstm'):
                action = predict_batched_value(policy,state,label)
            else:
                action = policy.predict(state)
        finally:
            np.random.set_state(rng_state)
        decision_times.append((time.perf_counter() - started) * 1000.0)
        velocity = np.asarray([action.vx, action.vy], dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if not np.all(np.isfinite(velocity)):
            raise RuntimeError('nonfinite baseline action')
        if speed > 1.0:
            velocity /= speed
        previous = np.asarray(env.robot.get_position(), dtype=np.float64)
        human_start = np.asarray([h.get_position() for h in env.humans])
        radii = np.asarray([h.radius + env.robot.radius for h in env.humans])
        pipeline_times.append((time.perf_counter() - cycle_start) * 1000.)
        _, _, terminated, truncated, info = env.step(
            action_cls(float(velocity[0]), float(velocity[1]))
        )
        current = np.asarray(env.robot.get_position(), dtype=np.float64)
        clearance = swept_min_clearance(previous, current, human_start,
                                        np.asarray([h.get_position() for h in env.humans]), radii)
        actual_minimum = min(actual_minimum, clearance)
        if clearance < -1e-9:
            overlap_steps += 1
            if first_overlap is None:
                first_overlap = float(env.global_time)
        path_length += float(np.linalg.norm(current - previous))
        minimum_clearance = min(minimum_clearance, float(info.get("dmin", math.inf)))
        event = str(info.get("event", "nothing"))

    success = int(event == "reach_goal")
    collision = int(event == "collision")
    timeout = int(event == "timeout")
    return EpisodeResult(
        arm=label,
        scenario=env.test_sim,
        human_num=env.human_num,
        case_id=case_id,
        event=event,
        success=success,
        collision=collision,
        timeout=timeout,
        nav_time=float(env.global_time),
        scored_time=float(env.global_time if success else env.time_limit),
        path_length=path_length,
        min_clearance=minimum_clearance,
        mean_plan_ms=float(np.mean(decision_times)),
        p95_plan_ms=float(np.percentile(decision_times, 95)),
        solver_steps=len(decision_times),
        observation_hash=hashlib.sha256("|".join(hashes).encode()).hexdigest(),
        actual_min_clearance=actual_minimum,
        actual_overlap_steps=overlap_steps,
        first_actual_overlap_time=first_overlap,
        collision_union=int(collision or overlap_steps > 0),
        success_without_overlap=int(success and overlap_steps == 0),
        mean_pipeline_ms=float(np.mean(pipeline_times)),
        p95_pipeline_ms=float(np.percentile(pipeline_times, 95)),
        deadline_miss_fraction=float(np.mean(np.asarray(pipeline_times) > env.time_step * 1000)),
        plan_step_ms=decision_times,
        pipeline_step_ms=pipeline_times,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("orca", "mamba", "cadrl", "sarl", "lstm", "dsrnn", "attngraph"), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--attngraph-root', type=Path, default=Path('/home/abc/temp/CrowdNav_Prediction_AttnGraph'))
    parser.add_argument('--no-occlusion', action='store_true')
    parser.add_argument("--crowdnav-root", type=Path, default=DEFAULT_CROWDNAV)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--human-count", type=int, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--circle-radius", type=float, default=None)
    parser.add_argument("--square-width", type=float, default=None)
    parser.add_argument("--time-limit", type=float, default=25.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--case-offset", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.controller == 'dsrnn' and not args.no_occlusion:
        raise ValueError('DSRNN missing-person semantics require a separately audited adapter')

    root_text = str(args.crowdnav_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    env, env_config, source = build_env(
        args.crowdnav_root,
        "sensor",
        args.human_count,
        args.scenario,
        args.circle_radius,
        args.square_width,
        occlusion=not args.no_occlusion,
    )
    env.time_limit = float(args.time_limit)
    env_config.set("env", "time_limit", str(args.time_limit))
    env_config.set("robot", "v_pref", "1.0")
    env.robot.v_pref = 1.0
    _, _, ActionXY, _ = _load_modules(args.crowdnav_root)
    defaults = {
        'cadrl':'crowd_nav/runs/mamba_vl/rl_model-cadrl.pth',
        'lstm':'crowd_nav/runs/mamba_vl/rl_model_lstm.pth',
        'sarl':'crowd_nav/runs/mamba_vl/rl_model_sarl.pth',
        'dsrnn':'crowd_nav/runs/dsrnn/dsrnn_27776.pt',
        'mamba':'crowd_nav/runs/mamba_vl/rl_model_ep9000_t24.pth',
        'orca':'unused',
    }
    checkpoint = args.checkpoint or (
        args.attngraph_root/'trained_models/GST_predictor_rand/checkpoints/41665.pt'
        if args.controller == 'attngraph' else args.crowdnav_root/defaults[args.controller]
    )
    policy, device = load_controller(
        args.controller, args.crowdnav_root, env_config, checkpoint, args.device, args.attngraph_root
    )

    results = []
    started = time.time()
    for case_id in range(args.case_offset, args.case_offset + args.episodes):
        row = run_episode(env, policy, ActionXY, args.controller, case_id)
        results.append(row)
        print(
            f"controller={args.controller} case={case_id} event={row.event} "
            f"time={row.nav_time:.2f} decision={row.mean_plan_ms:.2f}ms",
            flush=True,
        )

    payload = {
        "protocol": {
            "controller": args.controller,
            "device": device,
            "robot_visible": False,
            "observation": "current_visible_pedestrians_only",
            "occlusion": not args.no_occlusion,
            "v_max": 1.0,
            "time_limit": args.time_limit,
            "case_ids": [args.case_offset, args.case_offset + args.episodes - 1],
            "geometry": {
                "scenario": args.scenario,
                "human_count": args.human_count,
                "circle_radius": args.circle_radius,
                "square_width": args.square_width,
            },
            "checkpoint": str(checkpoint) if args.controller != "orca" else None,
            "checkpoint_sha256": (
                hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if args.controller != "orca" else None
            ),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "elapsed_seconds": time.time() - started,
        },
        "episodes": [asdict(row) for row in results],
        "summary": summarize(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
