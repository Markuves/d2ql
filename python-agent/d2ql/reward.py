import numpy as np


class RewardManager:
    def __init__(self, config: dict, e_ref: float = 1.0, v_max: float = 1.0):
        self.config = config

        # Energy-only reward (Task 2): r_t = -energy_delta / E_step_max
        # E_step_max = sum(HOST_MAX_WATT) * 1.0s / 3600.0 (watt-hours)
        # Aligned with Java HOST_MAX_WATT values.
        host_max_watts = [250.0, 200.0, 300.0, 220.0]
        self.e_step_max = sum(host_max_watts) * 1.0 / 3600.0  # wh per 1-second step
        self.e_ref = self.e_step_max

        # Keep weights field for API compatibility but neutralized.
        self.w = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.weight_history = []

    def compute_step_reward(
        self,
        energy_delta: float,
        sla_violations_this_step: float = 0.0,
        host_cpu_utilizations: list = None,
    ) -> float:
        """
        Energy-only step reward (Task 2): r_t = -energy_delta / E_step_max.
        Bounds r_t in [-1, 0] by construction.
        Unused arguments kept for API compatibility with existing callers.
        """
        return float(-energy_delta / self.e_step_max)

    def update_weights(self, delta_perf: float = 0.0, delta_energy: float = 0.0, delta_cost: float = 0.0) -> None:
        """No-op: adaptive weights removed for energy-only reward (Task 2)."""
        self.weight_history.append({"w_energy": 1.0})

    def get_current_weights(self) -> dict:
        return {"w_sla": 0.0, "w_energy": 1.0, "w_cost": 0.0}
