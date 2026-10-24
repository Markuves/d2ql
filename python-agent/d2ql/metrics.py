"""Centralized TensorBoard logging for d2ql runs.

Scalar namespaces:

``episode/*``  per-episode business metrics (reward, makespan, energy, cost,
               epsilon) plus the episode averages of the training diagnostics
               (``avg_loss``, ``mean_q``).
``train/*``    per-update diagnostics, always on the global update counter
               (``DDQNAgent.total_steps``), never on an episode-local index.
``buffer/*``   replay buffer occupancy.
``eval/*``     held-out evaluation metrics, logged once when the run ends.
``perf/*``     deployment benchmarks (latency, throughput) of the trained model.

Every series uses a single monotone x-axis per tag; mixing step counters in one
tag makes the TensorBoard chart unreadable (see docs/backlog.md, LOG-01).
"""

from __future__ import annotations

import logging


class MetricsLogger:
    """Thin wrapper around ``SummaryWriter`` with one tag per magnitude."""

    def __init__(self, log_dir: str = "outputs/runs"):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1, max_queue=1)
        logging.getLogger(__name__).info("TensorBoard writer ready at %s", log_dir)

    # ------------------------------------------------------------------
    # Episode-level metrics
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        makespan: float,
        energy: float,
        cost: float,
        epsilon: float,
    ) -> None:
        """Log standard per-episode scalars."""
        self.writer.add_scalar("episode/total_reward", total_reward, episode)
        self.writer.add_scalar("episode/makespan", makespan, episode)
        self.writer.add_scalar("episode/energy", energy, episode)
        self.writer.add_scalar("episode/cost", cost, episode)
        self.writer.add_scalar("episode/epsilon", epsilon, episode)
        self.writer.flush()

    def log_episode_diagnostics(self, episode: int, avg_loss: float, mean_q: float) -> None:
        """Episode averages of the per-update diagnostics (x = episode index)."""
        self.writer.add_scalar("episode/avg_loss", avg_loss, episode)
        self.writer.add_scalar("episode/mean_q", mean_q, episode)

    # ------------------------------------------------------------------
    # Per-update diagnostics
    # ------------------------------------------------------------------

    def log_training(self, step: int, loss: float, mean_q: float) -> None:
        """Log per-update diagnostics on the global update counter.

        No flush here: this is called once per training step (~tens of
        thousands of times per run) and TensorBoard's async writer already
        flushes on its own cadence.
        """
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/mean_q", mean_q, step)

    def log_buffer(self, episode: int, buffer_size: int) -> None:
        """Log replay buffer occupancy."""
        self.writer.add_scalar("buffer/size", buffer_size, episode)
        self.writer.flush()

    # ------------------------------------------------------------------
    # End-of-run summaries
    # ------------------------------------------------------------------

    def log_eval(self, step: int, metrics: dict[str, float]) -> None:
        """Log held-out evaluation metrics under ``eval/<key>``."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"eval/{key}", float(value), step)
        self.writer.flush()

    def log_perf(self, step: int, metrics: dict[str, float]) -> None:
        """Log deployment benchmarks (latency, throughput) under ``perf/<key>``."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"perf/{key}", float(value), step)
        self.writer.flush()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
