"""Traditional load-balancing heuristics for the H5 baseline comparison.

Each heuristic maps the *same* CloudSim environment observation the DDQN agent
sees — ``obs = [cpu_util x N_hosts, ram_util x N_hosts, queue_depth]`` — to a
host index, with no learning. H5 runs them through the identical CloudSim
simulator and held-out workload used for the H4 agent, collecting the same
business metrics (makespan, SLA violations, energy, cost, reward), so the
learned policy can be compared against classic baselines.

Adding a heuristic is a one-liner: subclass ``HeuristicPolicy``, set ``name``,
implement ``select_action``, and register it in ``HEURISTICS``.
"""

from __future__ import annotations

import numpy as np


class HeuristicPolicy:
    """Base class: stateless-ish host selector driven only by the observation."""

    name = "base"

    def __init__(self, n_hosts: int, seed: int | None = None) -> None:
        self.n_hosts = int(n_hosts)
        self.seed = seed
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Called at the start of every episode (deterministic restart)."""
        self.rng = np.random.default_rng(self._base_seed)

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> int:
        raise NotImplementedError

    # -- observation slices -------------------------------------------------
    def _cpu_utils(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs[: self.n_hosts], dtype=np.float64)

    def _ram_utils(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs[self.n_hosts : 2 * self.n_hosts], dtype=np.float64)


class RoundRobinPolicy(HeuristicPolicy):
    """Assign cloudlets to hosts in a fixed rotation (no load awareness)."""

    name = "round_robin"

    def __init__(self, n_hosts: int, seed: int | None = None) -> None:
        super().__init__(n_hosts, seed)
        self._counter = 0

    def reset(self) -> None:
        super().reset()
        self._counter = 0

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> int:
        host = self._counter % self.n_hosts
        self._counter += 1
        return int(host)


class RandomPolicy(HeuristicPolicy):
    """Assign each cloudlet to a uniformly random host."""

    name = "random"

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> int:
        return int(self.rng.integers(self.n_hosts))


class LeastLoadedPolicy(HeuristicPolicy):
    """Least-Loaded: pick the host with the lowest CPU utilization."""

    name = "least_loaded"

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> int:
        return int(np.argmin(self._cpu_utils(obs)))


class LeastCombinedPolicy(HeuristicPolicy):
    """Least-Combined-Resource: minimise cpu_util + ram_util per host."""

    name = "least_combined"

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> int:
        combined = self._cpu_utils(obs) + self._ram_utils(obs)
        return int(np.argmin(combined))


# Registry: name -> class. Extend to add a new baseline.
HEURISTICS: dict[str, type[HeuristicPolicy]] = {
    cls.name: cls
    for cls in (
        RoundRobinPolicy,
        RandomPolicy,
        LeastLoadedPolicy,
        LeastCombinedPolicy,
    )
}


def available_heuristics() -> list[str]:
    return list(HEURISTICS)


def build_heuristic(name: str, n_hosts: int, seed: int | None = None) -> HeuristicPolicy:
    key = str(name).strip().lower()
    if key not in HEURISTICS:
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {', '.join(HEURISTICS)}"
        )
    return HEURISTICS[key](n_hosts=n_hosts, seed=seed)
