"""Early stopping for H4 capacity x precision sweeps.

Each (precision, hidden_size) combination is trained from scratch. We stop a
run as soon as the *evaluation* reward stops improving, instead of always
running the full ``n_episodes`` budget. The metric is the mean reward over a
short trailing window (``window_size`` episodes) so single-episode noise does
not trigger a stop. We keep the best window so far and, after ``patience``
consecutive evaluations without a new best, we halt the run and report the
best window's stats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopResult:
    """Outcome reported when a run is stopped (or the budget is exhausted)."""

    stopped_early: bool = False
    best_episode: int = 0
    best_window_reward: float = float("-inf")
    best_window: dict = field(default_factory=dict)
    episodes_trained: int = 0
    evaluations: int = 0
    reason: str = ""


class EarlyStopper:
    """Stop training once the trailing-window reward plateaus.

    Parameters
    ----------
    patience:
        Number of *evaluations* (not episodes) to wait for improvement before
        stopping. One evaluation happens every ``eval_every`` episodes.
    window_size:
        How many recent evaluations are averaged to form the comparison metric.
        Larger windows are more stable but slower to react.
    min_evaluations:
        Do not stop before this many evaluations, so the model has time to
        leave its warm-up / initial noise region.
    min_delta:
        Minimum improvement over the best window reward to count as "better".
    mode:
        ``"max"`` to maximize reward (H4 uses reward maximization).
    """

    def __init__(
        self,
        patience: int = 3,
        window_size: int = 3,
        min_evaluations: int = 4,
        min_delta: float = 1e-3,
        mode: str = "max",
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if min_evaluations < 1:
            raise ValueError("min_evaluations must be >= 1")
        self.patience = patience
        self.window_size = window_size
        self.min_evaluations = min_evaluations
        self.min_delta = min_delta
        self.mode = mode

        self.eval_rewards: list[float] = []
        self.eval_episodes: list[int] = []
        self.best_window_reward: float | None = None
        self.best_window: dict = {}
        self.best_episode: int = 0
        self.since_best: int = 0
        self.evaluations: int = 0

    def _window_mean(self) -> float:
        if not self.eval_rewards:
            return float("-inf") if self.mode == "max" else float("inf")
        return sum(self.eval_rewards[-self.window_size :]) / len(
            self.eval_rewards[-self.window_size :]
        )

    def _is_better(self, candidate: float) -> bool:
        if self.best_window_reward is None:
            return True
        if self.mode == "max":
            return candidate > self.best_window_reward + self.min_delta
        return candidate < self.best_window_reward - self.min_delta

    def update(self, episode: int, reward: float, extra: dict | None = None) -> None:
        """Record an evaluation. Call once per ``eval_every`` episodes."""
        self.evaluations += 1
        self.eval_rewards.append(float(reward))
        self.eval_episodes.append(int(episode))
        window_reward = self._window_mean()
        if self._is_better(window_reward):
            self.best_window_reward = window_reward
            self.best_window = dict(extra or {})
            self.best_window["mean_reward"] = window_reward
            self.best_episode = episode
            self.since_best = 0
        else:
            self.since_best += 1

    def should_stop(self) -> bool:
        """True once patience is exhausted after enough evaluations."""
        if self.evaluations < self.min_evaluations:
            return False
        if self.best_window_reward is None:
            return False
        return self.since_best >= self.patience

    def result(self, episodes_trained: int, stopped_early: bool) -> EarlyStopResult:
        reason = (
            "reward plateau detected (early stop)"
            if stopped_early
            else "episode budget exhausted"
        )
        return EarlyStopResult(
            stopped_early=stopped_early,
            best_episode=self.best_episode,
            best_window_reward=(
                self.best_window_reward
                if self.best_window_reward is not None
                else float("-inf")
            ),
            best_window=self.best_window,
            episodes_trained=episodes_trained,
            evaluations=self.evaluations,
            reason=reason,
        )
