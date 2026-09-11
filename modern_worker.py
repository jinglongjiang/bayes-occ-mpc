"""Adapters that let the modern MPC comparators run inside this project's
evaluation loop, on the same layouts and the same observations as everything
else.

Three of the four comparators cannot live in this process:

  * SH-MPC and T-MPC++ are C++ and hold a compiled acados solver, so they run
    as the `mpc_bridge` executable and are driven over its line protocol;
  * SICNav needs a different acados build and a different Python environment,
    so it runs as a subprocess of its own interpreter.

Both cases are hidden behind the same small interface, which is the one
`run_episode` already uses:

    controller.reset()
    speed, turn = controller.act(observation)

so no scoring, layout or observation code is duplicated per method.

Observation contract
--------------------
Every controller is handed exactly what the project's own adapter produced --
visible detections and legal history only.  Ground truth reaches the evaluator
for scoring and never crosses the process boundary: `to_request` reads only
fields of PlannerObservation, which is already the filtered view.  A controller
that wants pedestrian futures gets the same shared-belief predictions the
Bayesian planner uses, and that arm is labelled separately from an end-to-end
one, because the two answer different questions.
"""
from __future__ import annotations

import math
import os
import selectors
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


BUILD_ROOT = Path(os.environ.get(
    "MODERN_BUILD_ROOT", "/home/abc/workspace/bayes_occ_mpc/build_modern"))
ACADOS_MPC_PLANNER = Path(os.environ.get(
    "ACADOS_MPC_PLANNER",
    "/home/abc/workspace/bayes_occ_mpc/results/acados_v042_mpcplanner"))
GSL_PREFIX = Path(os.environ.get(
    "GSL_PREFIX", "/home/abc/workspace/bayes_occ_mpc/build_modern/gsl"))
ROS_SETUP = Path(os.environ.get("ROS_SETUP", "/opt/ros/noetic/setup.bash"))


@dataclass
class StepReport:
    """What a controller did on one step, for the audit trail."""

    success: bool
    speed: float
    turn: float
    solve_ms: float
    total_ms: float
    exit_code: int


class BridgeController:
    """Drives `mpc_bridge`, which holds the upstream MPCPlanner::Planner.

    One process per episode arm, kept alive across steps so the planner's warm
    start behaves as it does natively.  The bridge writes its replies to a
    private descriptor, so ROS logging on stdout cannot corrupt them.
    """

    def __init__(self, variant: str, horizon: int, dt: float,
                 uncertainty_rate: float = 0.0, ros_master: Optional[str] = None,
                 settings: Optional[str] = None, human_margin: float = 0.0,
                 speed_limit: float = 1.0, path_overshoot: Optional[float] = None):
        # Diagnostic workspaces (tmpc_native, tmpc_nodummy, ...) are the same
        # planner built from a different configuration, so they share the base
        # variant's protocol semantics.  Keeping them nameable here is what makes
        # a one-variable-at-a-time comparison possible.
        base = "shmpc" if variant.startswith("shmpc") else (
            "tmpc" if variant.startswith("tmpc") else None)
        if base is None:
            raise ValueError(f"unknown bridge variant {variant!r}")
        self.base_variant = base
        self.variant = variant
        self.horizon = horizon
        self.dt = dt
        self.uncertainty_rate = uncertainty_rate
        # The Bayesian planner keeps r_robot + r_human + human_margin; these
        # planners keep r_robot + r_obstacle and ride exactly on that boundary,
        # so an episode grazes at precisely the distance the collision test
        # calls a collision.  Measured on a single static pedestrian with no
        # uncertainty: clearance settled at -0.0026 m, i.e. distance 0.5974
        # against the 0.6 the constraint enforces.  Folding the same margin into
        # the radius we send equalises the geometric budget; leaving it out
        # would score them against a tighter constraint than ours.
        self.human_margin = float(human_margin)
        # Reference-path sizing: the tracked window must outreach the horizon.
        self.speed_limit = float(speed_limit)
        # How far the reference continues past the goal.  This was a fixed 2.0 m
        # chosen by hand, which is not a defensible way to set a parameter that
        # decides whether the robot passes through the goal region or stops
        # short of it.  A traced episode showed 2.0 m driving the robot straight
        # past the goal to the spline's end -- closest approach 0.47 m, never
        # inside 0.25 m -- while 0.0 m had it stabilise short.  It is therefore
        # a development-set choice like risk or horizon, swept and selected by
        # the registered rule rather than set here.
        self.path_overshoot = (self.PATH_OVERSHOOT if path_overshoot is None
                               else float(path_overshoot))
        self.path_points = None
        self.path_segment_length = None

        workspace = BUILD_ROOT / variant
        self.binary = workspace / "devel/lib/mpc_planner_bridge/mpc_bridge"
        # A per-risk-level settings copy lets each method's own operating point
        # be swept without touching the shared configuration.
        self.settings = (Path(settings) if settings else
                         workspace / "src/mpc_planner/mpc_planner_jackalsimulator"
                                     "/config/settings.yaml")
        for path in (self.binary, self.settings):
            if not path.exists():
                raise FileNotFoundError(f"{variant}: missing {path}")

        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
            str(ACADOS_MPC_PLANNER / "lib"), str(GSL_PREFIX / "lib"),
            str(workspace / "devel/lib"), env.get("LD_LIBRARY_PATH", "")]))
        env["ACADOS_SOURCE_DIR"] = str(ACADOS_MPC_PLANNER)
        if ros_master:
            env["ROS_MASTER_URI"] = ros_master

        # The bridge's own log is kept when a directory is given, so a crash can
        # be diagnosed instead of appearing only as a closed pipe.
        log_dir = os.environ.get("MODERN_BRIDGE_LOG_DIR")
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._log = open(Path(log_dir) / f"{variant}_{os.getpid()}_{time.time_ns()}.log", "w")
        else:
            self._log = subprocess.DEVNULL
        self.process = subprocess.Popen(
            [str(self.binary), str(self.settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, bufsize=1, env=env)
        self._reply_buffer = bytearray()
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

        info = self._exchange("INFO").split()
        if info[0] != "INFO":
            raise RuntimeError(f"{variant}: unexpected INFO reply {info}")
        self.solver_horizon = int(info[1])
        self.solver_dt = float(info[2])
        self.max_obstacles = int(info[3])
        # Older bridges did not report the segment count; treat it as unknown
        # rather than guessing a value that silently shortens the tracked path.
        self.num_segments = int(info[5]) if len(info) > 5 and info[5].isdigit() else None
        # A mismatch here is a silent unfairness, not a warning: the solver was
        # generated for a horizon the caller is not using.
        if abs(self.solver_dt - dt) > 1e-9 or self.solver_horizon != horizon:
            raise RuntimeError(
                f"{variant}: solver was generated for N={self.solver_horizon} "
                f"dt={self.solver_dt}, evaluation asks for N={horizon} dt={dt}; "
                "regenerate the solver rather than running mismatched")

        self.last: Optional[StepReport] = None
        self.dropped_obstacles = 0
        self._path: Optional[List[Sequence[float]]] = None

    # ------------------------------------------------------------------ ipc

    def _exchange(self, line: str) -> str:
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.variant}: bridge exited "
                               f"with code {self.process.returncode}")
        deadline = time.monotonic() + float(os.environ.get("MODERN_IPC_TIMEOUT_S", "120"))
        payload = memoryview((line + "\n").encode())
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while payload or b"\n" not in self._reply_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.process.kill()
                    raise TimeoutError(f"{self.variant}: IPC timeout on {line.split()[0]}")
                for key, _ in selector.select(remaining):
                    if key.fileobj is self.process.stdin:
                        count = os.write(key.fd, payload[:65536])
                        payload = payload[count:]
                        if not payload:
                            selector.unregister(self.process.stdin)
                    else:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            raise RuntimeError(f"{self.variant}: bridge closed the connection")
                        self._reply_buffer.extend(chunk)
                        if len(self._reply_buffer) > 8_000_000:
                            raise RuntimeError(f"{self.variant}: oversized IPC reply")
        raw, _, remainder = self._reply_buffer.partition(b"\n")
        self._reply_buffer = bytearray(remainder)
        reply = raw.decode().strip()
        if not reply:
            raise RuntimeError(f"{self.variant}: bridge closed the connection")
        if reply.startswith("ERR "):
            raise RuntimeError(f"{self.variant}: {reply[4:]}")
        return reply

    def reset(self) -> None:
        self._exchange("RESET")
        self.dropped_obstacles = 0
        self._path = None

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._exchange("QUIT")
            except Exception:
                self.process.kill()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        if hasattr(self._log, "close"):
            self._log.close()

    # -------------------------------------------------------------- request

    def _reference_path(self, obs, points: Optional[int] = None) -> List[Sequence[float]]:
        """A straight line from the episode's start to its goal, built once.

        The Contouring module tracks progress along this path with a spline
        state.  Rebuilding the path from the robot's current position on every
        step resets that progress each time, so the controller never advances
        along its own reference -- which is what a per-step path did here.  The
        path is therefore fixed at the first step after a reset and reused, the
        way a roadmap reference behaves upstream.

        The crossing scenarios carry no map, so a straight start-to-goal line is
        the only reference available that does not encode where the pedestrians
        are.
        """
        if self._path is None:
            start, goal = obs.robot_xy, obs.goal_xy
            direction = goal - obs.robot_xy
            span = float(np.hypot(direction[0], direction[1]))
            # Optionally continue the line past the goal.  A contouring
            # controller stabilises onto the end of its spline, which leaves the
            # robot short of a 0.25 m ball; continuing the line makes it drive
            # through the goal region instead.  Neither is free: with the
            # native contour weight (0.05 against a lag weight of 0.75) the
            # robot tracks the line with several tenths of a metre of lateral
            # error, so a pass-through can miss the ball on the wrong side.
            # The amount is a swept parameter, not a constant chosen here.
            if span > 1e-9:
                goal = goal + direction / span * self.path_overshoot
            length = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
            if points is None:
                # The contouring cost only tracks `num_segments` spline pieces
                # ahead of the closest one; past that the spline extrapolates as
                # a constant, so the robot is asked to stop at the window's end.
                # Size the segments so that window reaches beyond what the
                # horizon can travel, with margin.
                reach = self.horizon * self.dt * self.speed_limit * self.PATH_WINDOW_MARGIN
                segments = self.num_segments or 5
                segment_length = max(reach / segments, 1e-3)
                points = max(4, int(np.ceil(length / segment_length)) + 1)
            self._path = [(start[0] + (goal[0] - start[0]) * i / (points - 1),
                           start[1] + (goal[1] - start[1]) * i / (points - 1))
                          for i in range(points)]
            self.path_points = points
            self.path_segment_length = length / max(points - 1, 1)
        return self._path

    # The two consumers read the same protocol field with opposite meanings.
    #
    #   T-MPC++  mpc_planner_modules/src/ellipsoid_constraints.cpp
    #            uses major/minor at step k as the MARGINAL semi-axes, scaled by
    #            its own chi(risk).
    #
    #   SH-MPC   scenario_module/src/sampler.cpp:452
    #            builds A_k = chol(Sigma_k) from the same field, then samples
    #                d_k = dt * sum_{j<=k} A_j xi_j
    #            i.e. it treats the field as a per-step process-noise INCREMENT
    #            and integrates it into a random walk (its own comment says the
    #            covariance "is assumed constant over the horizon").
    #
    # Sending one array to both would give SH-MPC a distribution nobody chose,
    # so each gets the form its own consumer expects, derived from the same
    # posterior.  verify_uncertainty.py checks the conversion against a known
    # marginal by Monte Carlo.
    UNCERTAINTY_SEMANTICS = {"tmpc": "marginal", "shmpc": "increment"}

    # SH-MPC doubles the configured risk (scenario_module/src/config.cpp:27,
    # `risk_ = CONFIG["probabilistic"]["risk"] * 2`); T-MPC++ does not.  The same
    # yaml number therefore means different things, and a swept value above 0.5
    # is out of range for SH-MPC rather than merely permissive.
    RISK_MULTIPLIER = {"tmpc": 1.0, "shmpc": 2.0}

    # How much further than the horizon can travel the tracked spline window
    # should reach.  1.0 would put the window end exactly at the horizon's reach.
    PATH_WINDOW_MARGIN = 1.5

    # How far the reference is continued past the goal, in metres.
    PATH_OVERSHOOT = 2.0

    @staticmethod
    def _to_ellipse(covariance):
        values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
        values = np.maximum(values, 0.0)
        minor = np.sqrt(values[..., 0])
        major = np.sqrt(values[..., 1])
        principal = vectors[..., 1]               # eigenvector of the larger value
        tilt = np.arctan2(principal[..., 1], principal[..., 0])
        return major, minor, tilt

    def _ellipses(self, obs):
        """The project's posterior in the form this consumer actually reads.

        SH-MPC asserts that predictions carry non-zero uncertainty and refuses a
        point prediction outright, so the belief has to be shared.  The arm is
        labelled shared-belief, not end-to-end: it is not the comparator's own
        inference.
        """
        covariance = obs.human_position_covariance
        if covariance is None:
            return None
        covariance = np.asarray(covariance, dtype=np.float64)   # (H, N, 2, 2)
        if covariance.size == 0:
            # Occlusion can leave a step with nothing visible.  Reducing over an
            # empty array raised here, which killed the SH-MPC arm outright on
            # any sparse scene -- an interfacing fault, not the method declining
            # to plan.  An empty detection set is a legal observation.
            return None

        if self.UNCERTAINTY_SEMANTICS[self.base_variant] == "increment":
            # Cov(d_k) = dt^2 * sum_{j<=k} S_j, so S_k = (P_k - P_{k-1}) / dt^2.
            shifted = np.concatenate(
                (np.zeros_like(covariance[:, :1]), covariance[:, :-1]), axis=1)
            covariance = (covariance - shifted) / (self.dt * self.dt)
            smallest = float(np.linalg.eigvalsh(covariance).min())
            if smallest < -1e-9:
                # A shrinking marginal cannot be expressed as a random walk.
                # Reporting beats clipping, which would silently change the
                # distribution the planner is given.
                raise RuntimeError(
                    f"{self.variant}: posterior covariance decreases between "
                    f"horizon steps (min increment eigenvalue {smallest:.3e}); "
                    "it cannot be expressed as the random walk this sampler "
                    "integrates")
            self.negative_increments = max(getattr(self, "negative_increments", 0.0),
                                           -min(0.0, smallest))
        return self._to_ellipse(covariance)

    def to_request(self, obs) -> str:
        """Serialise one step.  Reads only PlannerObservation, which is already
        the filtered, visible-only view."""
        if obs.robot_heading is None:
            raise ValueError("bridge controller needs observation.robot_heading")

        parts = ["STEP",
                 f"{obs.robot_xy[0]:.9f}", f"{obs.robot_xy[1]:.9f}",
                 f"{float(obs.robot_heading):.9f}",
                 f"{float(np.linalg.norm(obs.robot_velocity)):.9f}",
                 f"{obs.goal_xy[0]:.9f}", f"{obs.goal_xy[1]:.9f}"]

        path = self._reference_path(obs)
        parts += ["NPATH", str(len(path))]
        for px, py in path:
            parts += [f"{px:.9f}", f"{py:.9f}"]

        entities = np.asarray(obs.entities, dtype=np.float64).reshape(-1, 5)
        if entities.shape[0] > self.max_obstacles:
            # Never silently drop: the caller must see it and the run is invalid.
            self.dropped_obstacles = entities.shape[0] - self.max_obstacles
            raise RuntimeError(
                f"{self.variant}: {entities.shape[0]} detections exceed the "
                f"solver's {self.max_obstacles} obstacle slots")

        # Prefer the project's own predicted segments when they are present, so
        # every method is judged on the same future; fall back to the constant
        # velocity implied by the detection when they are not.
        starts = obs.human_segment_start
        ends = obs.human_segment_end
        buffer = obs.human_uncertainty_buffer
        ellipse = self._ellipses(obs)

        parts += ["NOBS", str(entities.shape[0])]
        for i, row in enumerate(entities):
            px, py, vx, vy, radius = row
            radius = radius + self.human_margin
            angle = math.atan2(vy, vx) if (vx or vy) else 0.0
            parts += [str(i), f"{px:.9f}", f"{py:.9f}", f"{angle:.9f}",
                      f"{radius:.9f}", "NPRED", str(self.horizon)]
            for k in range(self.horizon):
                if starts is not None and ends is not None:
                    mx, my = ends[i, k, 0], ends[i, k, 1]
                else:
                    mx = px + vx * (k + 1) * self.dt
                    my = py + vy * (k + 1) * self.dt
                if ellipse is not None:
                    major, minor, tilt = (float(ellipse[0][i, k]),
                                          float(ellipse[1][i, k]),
                                          float(ellipse[2][i, k]))
                elif buffer is not None:
                    major = minor = float(buffer[i, k])
                    tilt = angle
                elif self.uncertainty_rate:
                    major = minor = self.uncertainty_rate * (k + 1) * self.dt
                    tilt = angle
                else:
                    major = minor = 0.0
                    tilt = angle
                parts += [f"{mx:.9f}", f"{my:.9f}", f"{tilt:.9f}",
                          f"{major:.9f}", f"{minor:.9f}"]
        return " ".join(parts)

    def act(self, obs):
        """Returns (speed, heading increment), the unicycle command CrowdSim
        consumes, plus the plan time in milliseconds."""
        start = time.perf_counter()
        reply = self._exchange(self.to_request(obs)).split()
        if reply[0] != "STEP":
            raise RuntimeError(f"{self.variant}: unexpected reply {reply}")
        report = StepReport(
            success=bool(int(reply[1])), speed=float(reply[2]),
            turn=float(reply[3]) * self.dt,   # the bridge returns an angular rate
            solve_ms=float(reply[4]), total_ms=float(reply[5]),
            exit_code=int(reply[6]))
        self.last = report
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return np.array([report.speed, report.turn], dtype=np.float64), elapsed_ms


def bridge_planner_factory(variant: str, uncertainty_rate: float = 0.0,
                           ros_master: Optional[str] = None,
                           settings: Optional[str] = None,
                           path_overshoot: Optional[float] = None):
    """A `planner_type` for run_episode: constructed with the MPCConfig, and
    exposing the same `plan(observation, seed)` the CEM planner does.

    The seed is accepted but cannot be honoured.  SH-MPC's scenario sampler
    draws from `std::mt19937{std::random_device{}()}`
    (scenario_module/src/sampler.cpp:377), so its scenarios differ run to run and
    the arm is stochastic no matter what is passed here.  That is reported as a
    property of the method rather than papered over; repeated runs are the only
    way to bound it.
    """

    class _Factory:
        def __init__(self, config):
            self._controller = BridgeController(
                variant, config.horizon, config.dt,
                uncertainty_rate=uncertainty_rate, ros_master=ros_master,
                settings=settings,
                human_margin=getattr(config, "human_margin", 0.0),
                speed_limit=getattr(config, "v_max", 1.0),
                path_overshoot=path_overshoot)
            self.last_diagnostics = {}

        def reset(self):
            self._controller.reset()

        def plan(self, observation, seed):
            action, elapsed_ms = self._controller.act(observation)
            report = self._controller.last
            self.last_diagnostics = {
                "solver_success": float(report.success),
                "solve_ms": report.solve_ms,
                "exit_code": float(report.exit_code),
            }
            return action, elapsed_ms

        def __del__(self):
            try:
                self._controller.close()
            except Exception:
                pass

    _Factory.__name__ = f"Bridge_{variant}"
    return _Factory
