import logging
import random
import socket
import time
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from py4j.protocol import Py4JNetworkError

from d2ql.queue import PriorityCloudletQueue
from d2ql.reward import RewardManager

logger = logging.getLogger(__name__)

# The Java gateway builds a fixed topology (SimulationGateway.NUM_HOSTS = 4).
# That used to be configurable on the Python side only, so any other value
# silently produced an observation/action space that did not match the
# simulator. It is now a single constant mirrored here.
N_HOSTS = 4
# obs = [cpu_util x N_HOSTS, ram_util x N_HOSTS, queue_depth, mi_norm, pes_norm]
# The gateway still emits a 12th element (normalised time-to-deadline) but it is
# always 1.0: trace deadlines are absolute timestamps ~1e6 s away from a clock
# that runs ~170 s, so it carries no information and is dropped here (see
# docs/backlog.md, SIM-01). Cloudlet level is included because it changes the
# dispatch cost.
OBS_DIM = N_HOSTS * 2 + 1 + 2

# ---------------------------------------------------------------------------
# Py4J gateway resilience
# ---------------------------------------------------------------------------
# java-sim is restarted by Docker (OOM, new image, host reboot). A py4j call made
# while it is down fails with a socket error rather than with a simulation error,
# and those failures used to escape and kill the training process (13 of the 65
# container restarts in the incident). They are retried with exponential backoff
# and jitter instead. Logic errors -- Java-side exceptions, bad arguments -- are
# never retried: they would fail identically on every attempt.
RETRY_INITIAL_DELAY_S = 1.0
RETRY_BACKOFF_FACTOR = 2.0
RETRY_MAX_DELAY_S = 30.0
RETRY_MAX_ATTEMPTS = 10
#: Each delay is multiplied by (1 +/- this), so clients do not retry in lockstep.
RETRY_JITTER_FRAC = 0.2

#: Errors that only mean "the gateway is not reachable right now". A py4j
#: "Answer from Java side is empty" is a Py4JNetworkError, so it lands here too.
TRANSIENT_GATEWAY_ERRORS = (socket.gaierror, ConnectionRefusedError, Py4JNetworkError)


class GatewayError(RuntimeError):
    """A Py4J gateway call that failed after retrying the transient errors.

    Every py4j call in this module goes through :func:`retry_call` and raises
    this single type, carrying the call site and the episode context, so callers
    never have to handle raw py4j exceptions. Handling used to be inconsistent:
    "Answer from Java side is empty" was swallowed in some paths and fatal in
    others.
    """

    def __init__(self, call_site: str, attempts: int, context: dict | None = None, cause=None):
        self.call_site = call_site
        self.attempts = attempts
        self.context = dict(context or {})
        self.cause = cause
        details = ", ".join(f"{key}={value}" for key, value in self.context.items())
        message = f"Py4J gateway call '{call_site}' failed after {attempts} attempt(s)"
        if details:
            message += f" ({details})"
        if cause is not None:
            message += f": {type(cause).__name__}: {cause}"
        super().__init__(message)


def retry_delay(attempt: int) -> float:
    """Delay before retry number ``attempt`` (1-based): 1 s, 2 s, 4 s ... max 30 s."""
    base = min(
        RETRY_INITIAL_DELAY_S * RETRY_BACKOFF_FACTOR ** (attempt - 1), RETRY_MAX_DELAY_S
    )
    return base * (1.0 + random.uniform(-RETRY_JITTER_FRAC, RETRY_JITTER_FRAC))


def retry_call(
    call_site: str, func: Callable[[], Any], context: dict | None = None
) -> Any:
    """Call ``func`` and retry it on transient gateway errors, with backoff.

    Returns the first successful result. Logs every retry at WARNING and, once
    the attempt budget is exhausted, raises :class:`GatewayError` with the call
    site and the ``context`` (episode, cloudlet count) instead of leaking the raw
    py4j exception. Non-transient errors propagate unchanged on the first try.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except TRANSIENT_GATEWAY_ERRORS as exc:
            if attempt >= RETRY_MAX_ATTEMPTS:
                raise GatewayError(call_site, attempt, context, exc) from exc
            delay = retry_delay(attempt)
            logger.warning(
                "Py4J call %s failed (attempt %d/%d): %s: %s - retrying in %.1fs",
                call_site,
                attempt,
                RETRY_MAX_ATTEMPTS,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)


class CloudSimEnv(gym.Env):
    """Gymnasium wrapper around the CloudSimPlus Java simulation via Py4J."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, reward_fn=None):
        super().__init__()
        self.config = config

        # Action: assign the next pending cloudlet to one of the 4 hosts.
        self.action_space = spaces.Discrete(N_HOSTS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Reward lives in the env so `step()` honours the Gymnasium contract
        # (SIM-05): r_t = -energy_delta / E_step_max, in [-1, 0].
        self.reward_fn = reward_fn if reward_fn is not None else RewardManager()

        # Dispatch order (LPT by work-per-PE); rebuilt on every reset.
        self.dispatch_queue = PriorityCloudletQueue()

        self.gateway = None
        self._prev_energy = 0.0  # cumulative energy of the previous step (A1)
        self._supports_episode_complete: bool | None = None
        self._episode = 0  # 1-based episode counter, for gateway error context
        self._n_cloudlets = 0
        self._steps = 0
        self._max_steps = 0
        self._connect_gateway()

    # ------------------------------------------------------------------
    # Py4J connection
    # ------------------------------------------------------------------

    def _connect_gateway(self):
        import os

        from py4j.java_gateway import GatewayParameters, JavaGateway

        py4j_cfg = self.config.get("py4j") or {}
        address = py4j_cfg.get("host") or os.environ.get("JAVA_HOST", "java-sim")
        port = int(py4j_cfg.get("port") or os.environ.get("JAVA_PORT", 25333))

        def _open():
            gateway = JavaGateway(
                gateway_parameters=GatewayParameters(address=address, port=port)
            )
            # entry_point is a round trip to the JVM: it fails too while
            # java-sim is still booting, so it belongs inside the retried block.
            return gateway, gateway.entry_point

        self.gateway, self.sim = retry_call(
            f"connect({address}:{port})", _open, {"address": address, "port": port}
        )

    def _gateway_call(self, name: str, *args) -> Any:
        """The single retried, uniformly handled path for every py4j call.

        FIX 4: transient socket errors while java-sim restarts are retried with
        backoff instead of killing training. FIX 5: every call site behaves the
        same way and reports failures with the episode context.
        """
        func = getattr(self.sim, name)
        context = {"episode": self._episode, "cloudlets": self._n_cloudlets}
        return retry_call(name, lambda: func(*args), context)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None, cloudlets=None):
        super().reset(seed=seed)

        if cloudlets is None and isinstance(options, dict):
            cloudlets = options.get("cloudlets")

        # FIX 5: episode context for gateway error messages. Set before the first
        # gateway call so a reset failure still reports the window it was loading.
        self._episode += 1
        self._n_cloudlets = 0 if not cloudlets else len(cloudlets)

        # Push experiment seed to the Java gateway (D3 fix), honouring reset(seed=).
        episode_seed = int(self.config.get("experiment", {}).get("seed", 42))
        if seed is not None:
            episode_seed = int(seed)
        self._gateway_call("setSeed", episode_seed)

        self.dispatch_queue = PriorityCloudletQueue()

        if cloudlets:
            # SIM-04: the queue now actually decides the dispatch order. The
            # gateway places cloudlets in the order it receives them, so we sort
            # the episode window here (LPT) instead of relying on trace order.
            by_id = {}
            for spec in cloudlets:
                cid = int(spec.cloudlet_id)
                by_id[cid] = spec
                self.dispatch_queue.push(
                    cloudlet_id=cid, mi=float(spec.mi), num_pes=int(spec.num_pes)
                )
            ordered_ids = self.dispatch_queue.drain()
            logger.debug(
                "dispatch order (LPT, work/PE desc): %s ...", ordered_ids[:10]
            )

            self._gateway_call("clearWorkload")
            for cid in ordered_ids:
                spec = by_id[cid]
                self._gateway_call(
                    "addWorkloadRow",
                    float(spec.submitted_at),
                    float(spec.deadline),
                    float(spec.mi),
                    float(spec.num_pes),
                )
            raw = self._gateway_call("resetEpisode")
        else:
            raw = self._gateway_call("reset")

        self._prev_energy = 0.0  # A1: each episode's energy delta starts at zero
        self._steps = 0
        # Hard cap: 1 dispatch per step + drain, with slack. Prevents a hang if
        # the engine stops producing events mid-window (SIM-10).
        self._max_steps = max(20 * self._n_cloudlets + 100, 200)
        obs = self._parse_obs(raw)
        info = {"n_cloudlets": self._n_cloudlets}
        return obs, info

    def step(self, action: int):
        host_idx = int(action) % N_HOSTS
        self._steps += 1

        # Execute the action in the Java simulator
        raw = self._gateway_call("step", host_idx)
        obs = self._parse_obs(raw)

        # Gather step metrics from Java
        cpu_utilizations = list(self._gateway_call("getHostCpuUtilizations"))
        energy = float(self._gateway_call("getTotalEnergyConsumed"))
        makespan = float(self._gateway_call("getMakespan"))
        cost = float(self._gateway_call("getOperationalCost"))

        # A1: report the per-step energy increment, not the cumulative total, so
        # the reward term stays bounded and comparable across episode lengths.
        energy_delta = energy - self._prev_energy
        self._prev_energy = energy

        # SIM-08: the episode ends when the last cloudlet finishes, not when the
        # simulator's event queue drains. Two safety nets keep a run from
        # hanging when the engine runs out of events before the window is done
        # (see SIM-10): no running engine and a hard step cap.
        finished_cloudlets = self._finished_cloudlets()
        complete = bool(self._episode_complete())
        sim_running = self._simulation_running()
        terminated = complete
        truncated = False
        if not complete:
            if not sim_running:
                terminated = truncated = True
            elif self._steps >= self._max_steps:
                terminated = truncated = True

        info = {
            "cpu_utilizations": cpu_utilizations,
            "energy": energy,              # cumulative (episode totals / logs)
            "energy_delta": energy_delta,  # per-step increment (SIM-05)
            "makespan": makespan,
            "cost": cost,
            "sim_time": float(self._gateway_call("getSimulationTime")),  # clock AFTER the step
            "finished_cloudlets": finished_cloudlets,
            "n_cloudlets": self._n_cloudlets,
            "episode_complete": complete,
            "simulation_running": sim_running,
        }

        # SIM-05: the env owns the reward (r_t in [-1, 0]); callers just use the
        # value returned by step(), as Gymnasium expects.
        reward = self.reward_fn.compute_step_reward(energy_delta)
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        """Drop the client connection.

        ``JavaGateway.shutdown()`` would stop the **server**, and the gateway is
        shared by every episode/run in the process (and by H5 after H4), so the
        next ``reset`` used to fail with a dead socket. ``close()`` only closes
        the client channel.
        """
        if self.gateway is not None:
            try:
                self.gateway.close()
            except Exception:  # noqa: BLE001 - best effort teardown
                pass
            self.gateway = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episode_complete(self) -> bool:
        """SIM-08: all cloudlets dispatched and finished (gateway >= 07f07ccf).

        Falls back to ``isFinished()`` on older gateways, which waits for the
        simulator event queue to drain and therefore includes the idle tail.
        """
        if self._supports_episode_complete is None:
            try:
                self._gateway_call("isEpisodeComplete")
                self._supports_episode_complete = True
            except GatewayError:
                # The gateway is unreachable, not old: retrying already failed.
                raise
            except Exception:  # noqa: BLE001 - old gateway image
                self._supports_episode_complete = False
                logger.warning(
                    "Gateway has no isEpisodeComplete(); falling back to "
                    "isFinished() and episodes will include the idle tail (SIM-08)."
                )
        if self._supports_episode_complete:
            return bool(self._gateway_call("isEpisodeComplete"))
        return bool(self._gateway_call("isFinished"))

    def _finished_cloudlets(self) -> int:
        """Cloudlets of the window that already finished (-1 if unsupported)."""
        try:
            return int(self._gateway_call("getFinishedCloudletCount"))
        except GatewayError:
            raise
        except Exception:  # noqa: BLE001 - old gateway image
            return -1

    def _simulation_running(self) -> bool:
        """Whether the CloudSim engine still has events to process."""
        try:
            return bool(self._gateway_call("isSimulationRunning"))
        except GatewayError:
            raise
        except Exception:  # noqa: BLE001 - old gateway image
            return True

    def _parse_obs(self, raw) -> np.ndarray:
        """Convert the Java observation array to the env's float32 vector.

        The gateway emits ``N_HOSTS*2 + 1 + 3`` values; only the first
        ``OBS_DIM`` are kept (the trailing time-to-deadline is constant, see the
        module docstring).
        """
        try:
            obs = np.array(list(raw), dtype=np.float32)
        except Exception:
            obs = np.zeros(OBS_DIM, dtype=np.float32)

        if obs.size < OBS_DIM:
            obs = np.pad(obs, (0, OBS_DIM - obs.size))
        elif obs.size > OBS_DIM:
            obs = obs[:OBS_DIM]

        return np.clip(obs, 0.0, 1.0)
