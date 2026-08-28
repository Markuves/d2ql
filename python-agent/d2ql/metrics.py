from typing import Optional


class MetricsLogger:
    """Centralized TensorBoard logging wrapper for d2ql training runs."""

    def __init__(self, log_dir: str = "outputs/runs"):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.writer.add_scalar("run/started", 1.0, 0)
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
    ) -> None:
        """Log standard per-episode scalars."""
        self.writer.add_scalar("episode/total_reward", total_reward, episode)
        self.writer.add_scalar("episode/makespan", makespan, episode)
        self.writer.add_scalar("episode/energy", energy, episode)
        self.writer.add_scalar("episode/cost", cost, episode)
        self.writer.add_scalar("episode/epsilon", epsilon, episode)
        self.writer.flush()

    # ------------------------------------------------------------------
    # Adaptive weight metrics
    # ------------------------------------------------------------------

    def log_weights(
        self,
        episode: int,
        w_sla: float,
        w_energy: float,
        w_cost: float,
    ) -> None:
        """Log reward manager adaptive weights."""
        self.writer.add_scalar("weights/w_sla", w_sla, episode)
        self.writer.add_scalar("weights/w_energy", w_energy, episode)
        self.writer.add_scalar("weights/w_cost", w_cost, episode)
        self.writer.flush()

    # ------------------------------------------------------------------
    # Agent / training metrics
    # ------------------------------------------------------------------

    def log_training(
        self,
        step: int,
        loss: float,
        mean_q: float,
    ) -> None:
        """Log per-update training diagnostics."""
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/mean_q", mean_q, step)
        self.writer.flush()

    def log_buffer(self, episode: int, buffer_size: int) -> None:
        """Log replay buffer occupancy."""
        self.writer.add_scalar("buffer/size", buffer_size, episode)
        self.writer.flush()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
