"""Centralized TensorBoard logging for d2ql runs.

Scalar namespaces (each one uses a single, monotone x-axis):

``episode/*``  per-episode business metrics (reward, makespan, energy, cost,
               epsilon) plus ``steps``, ``dynamic_energy`` and the episode
               averages of the training diagnostics.
``train/*``    per-update diagnostics on the global update counter
               (``DDQNAgent.total_steps``): loss, mean Q, grad norm, |TD error|, lr.
``buffer/*``   replay buffer occupancy.
``stopper/*``  the trailing-window reward that decides early stopping, at the
               episode index where it is evaluated.
``eval/*``     held-out evaluation metrics, once at the end of the run.
``perf/*``     deployment benchmarks (latency, throughput) of the trained model.
``run/*``      text metadata (e.g. why the run stopped).

Values that are not finite (NaN/inf) are never written: a missing measurement
must not appear as a data point (docs/backlog.md, LOG-06).
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Thin wrapper around ``SummaryWriter`` with one tag per magnitude."""

    def __init__(self, log_dir: str = "outputs/runs"):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1, max_queue=1)
        logger.info("TensorBoard writer ready at %s", log_dir)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write one finite scalar (silently skipped when NaN/inf)."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self.writer.add_scalar(tag, value, int(step))

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Write a text event (e.g. the early-stopping reason)."""
        self.writer.add_text(tag, str(text), int(step))

    def log_scalars(self, prefix: str, metrics: dict[str, float], step: int) -> None:
        """Write a batch of scalars under ``prefix/<key>``, skipping non-finite."""
        for key, value in metrics.items():
            self.log_scalar(f"{prefix}/{key}", value, step)
        self.writer.flush()

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
        steps: int | None = None,
        dynamic_energy: float | None = None,
    ) -> None:
        """Log per-episode business metrics.

        ``steps`` and ``dynamic_energy`` are what makes the reward readable:
        without them the reward curve is mostly episode length (LOG-05/LOG-07).
        """
        self.log_scalar("episode/total_reward", total_reward, episode)
        self.log_scalar("episode/makespan", makespan, episode)
        self.log_scalar("episode/energy", energy, episode)
        self.log_scalar("episode/cost", cost, episode)
        self.log_scalar("episode/epsilon", epsilon, episode)
        if steps is not None:
            self.log_scalar("episode/steps", steps, episode)
            if float(steps) > 0:
                self.log_scalar("episode/reward_per_step", total_reward / float(steps), episode)
                self.log_scalar("episode/energy_per_step", energy / float(steps), episode)
        if dynamic_energy is not None:
            self.log_scalar("episode/dynamic_energy", dynamic_energy, episode)
        self.writer.flush()

    def log_episode_diagnostics(
        self, episode: int, avg_loss: float, mean_q: float, updates: int
    ) -> None:
        """Episode averages over the **updates** of that episode (not env steps).

        Dividing by env steps used to deflate the first episode by ~1.7x because
        the replay warm-up contributes zeros (LOG-04); with no updates there is
        nothing to average and no point is written.
        """
        if updates <= 0:
            return
        self.log_scalar("episode/avg_loss", avg_loss, episode)
        self.log_scalar("episode/mean_q", mean_q, episode)

    # ------------------------------------------------------------------
    # Per-update diagnostics
    # ------------------------------------------------------------------

    def log_training(
        self,
        step: int,
        loss: float,
        mean_q: float,
        grad_norm: float | None = None,
        td_error_abs: float | None = None,
        lr: float | None = None,
    ) -> None:
        """Log per-update diagnostics on the global update counter.

        No flush here: this runs once per training step (tens of thousands of
        times per run) and TensorBoard's async writer flushes on its own cadence.
        """
        self.log_scalar("train/loss", loss, step)
        self.log_scalar("train/mean_q", mean_q, step)
        if grad_norm is not None:
            self.log_scalar("train/grad_norm", grad_norm, step)
        if td_error_abs is not None:
            self.log_scalar("train/td_error_abs", td_error_abs, step)
        if lr is not None:
            self.log_scalar("train/lr", lr, step)

    def log_buffer(self, episode: int, buffer_size: int) -> None:
        """Log replay buffer occupancy."""
        self.log_scalar("buffer/size", buffer_size, episode)
        self.writer.flush()

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def log_stopper(self, episode: int, window_mean: float, best_window_mean: float) -> None:
        """Log the trailing-window reward that decides early stopping."""
        self.log_scalar("stopper/window_mean", window_mean, episode)
        self.log_scalar("stopper/best_window_mean", best_window_mean, episode)

    # ------------------------------------------------------------------
    # End-of-run summaries
    # ------------------------------------------------------------------

    def log_eval(self, step: int, metrics: dict[str, float]) -> None:
        """Log held-out evaluation metrics under ``eval/<key>``."""
        self.log_scalars("eval", metrics, step)

    def log_perf(self, step: int, metrics: dict[str, float]) -> None:
        """Log deployment benchmarks (latency, throughput) under ``perf/<key>``."""
        self.log_scalars("perf", metrics, step)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
