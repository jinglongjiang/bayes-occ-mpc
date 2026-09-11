"""A restricted view of the simulator for policies that read history themselves.

SICNav-Diffusion warms its predictor by reading the simulator's own state log
directly:

    sicnav_diffusion/policy/sicnav_acados.py:1177
        states_history = self.env.states[-num_hist_frames-1:-1]

`env.states` is ground truth.  Under full observation that is not a violation --
the policy could have seen all of it -- but under this project's occlusion
protocol it hands the predictor the positions of pedestrians the robot cannot
see, including ones that were never detected.  A run like that would measure the
predictor against information no deployed system has.

This facade stands in for the environment.  It forwards the attributes the
policy legitimately needs (its own clock, its own configuration, the scenario
geometry) and replaces `states` with a history the adapter maintains from
detections only:

  * a pedestrian that was not detected at a timestep is absent from that frame,
    not back-filled with where it actually was;
  * a pedestrian that reappears keeps its detection identity, and the gap stays
    a gap;
  * the robot's own state is its own proprioception and is passed through.

If the policy asks for something the facade does not offer, that is an access we
have not thought about, so it raises rather than silently forwarding to the real
environment.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


# Attributes the policy may read: its own clock, its own configuration, and the
# scenario geometry, none of which is a pedestrian observation.
_FORWARDED = frozenset({
    "config", "global_time", "global_time_step", "time_step", "time_limit",
    "sim_env", "robot", "square_width", "circle_radius",
    "door_x_mid", "door_y_min", "door_y_max", "door_y_mid_min", "door_y_mid_max",
})


class VisibleHistoryEnv:
    """Wraps a CrowdSim environment and rewrites its history.

    `history` is a list of (self_state, human_states) frames, oldest first, built
    by the adapter from what the robot actually detected.  It is what
    `env.states` returns.
    """

    def __init__(self, env, history: Optional[List[Tuple[Any, Sequence[Any]]]] = None):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_history", list(history or []))

    # ------------------------------------------------------------- history

    @property
    def states(self):
        return self._history

    def append_frame(self, self_state, human_states) -> None:
        """Record one timestep of legal observation.

        `human_states` must already be the detected subset; passing the full
        crowd here would reintroduce exactly the leak this class exists to
        close, so the caller is the adapter, never the environment.
        """
        self._history.append((self_state, list(human_states)))

    def reset_history(self) -> None:
        self._history.clear()

    def frames_available(self) -> int:
        return len(self._history)

    def has_enough_history(self, required: int) -> bool:
        """Whether the predictor's warm-up can be satisfied legally.

        The alternative -- letting it read past the start of the log -- is what
        raises IndexError inside the policy, and the tempting fix there is to
        pad with truth.  Callers should check this and skip or wait instead.
        """
        return len(self._history) >= required + 1

    # ------------------------------------------------------- attribute wall

    def __getattr__(self, name: str):
        if name in _FORWARDED:
            return getattr(self._env, name)
        raise AttributeError(
            f"{name!r} is not exposed by VisibleHistoryEnv. The policy is "
            f"reaching into the simulator for something that has not been "
            f"reviewed for observability; add it to the forwarded set only "
            f"after checking it carries no hidden pedestrian state."
        )

    def __setattr__(self, name: str, value):
        raise AttributeError(
            "VisibleHistoryEnv is read-only; a policy must not write to the "
            "environment through it"
        )
