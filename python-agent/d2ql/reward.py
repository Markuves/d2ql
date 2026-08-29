import numpy as np


class RewardManager:
    def __init__(self, config: dict, e_ref: float = 1.0, v_max: float = 1.0):
        self.config = config

        # Calibration baselines
        self.e_ref = e_ref
        self.v_max = v_max

        # Hyperparameters
        self.target_util = config["reward"]["target_utilization"] # default 0.70
        self.alpha_w = config["reward"]["weight_lr"] # default 0.01

        # Weight floors
        self.w_sla_floor = config["reward"]["w_perf_floor"] # default 0.2
        self.w_energy_floor = config["reward"]["w_energy_floor"] # default 0.1
        self.w_cost_floor = config["reward"]["w_cost_floor"] # default 0.1
        self.sum_floors = self.w_sla_floor + self.w_energy_floor + self.w_cost_floor # 0.4

        # Current weights (initialized to defaults)
        self.w = np.array([
            config["reward"]["w_perf_init"], # default 0.4
            config["reward"]["w_energy_init"], # default 0.3
            config["reward"]["w_cost_init"]  # default 0.3
        ], dtype=np.float32)

        self.weight_history = [] # records (w_sla, w_energy, w_util) per episode

    def compute_step_reward(
        self,
        energy_delta: float,
        sla_violations_this_step: float,
        host_cpu_utilizations: list,
    ) -> float:
        """
        Computes the composite step reward r_t based on current metric weights.

        `energy_delta` is the *increment* of energy consumed by THIS step
        (E_t - E_{t-1}), not the cumulative total. This keeps the energy term
        bounded and comparable across episodes of different lengths (A1 fix).
        """
        # 1. Energy Component: phi_energy(t) = -(E_t - E_{t-1}) / E_ref
        phi_energy = -energy_delta / self.e_ref

        # 2. SLA Component: phi_sla(t) = -indicator(v_t > 0) * v_t / v_max
        indicator = 1.0 if sla_violations_this_step > 0 else 0.0
        phi_sla = -indicator * (sla_violations_this_step / self.v_max)

        # 3. Utilization Component: phi_util(t) = 1 - (1/N) * sum_i |u_cpu_i_t - u*|
        n_hosts = len(host_cpu_utilizations)
        if n_hosts > 0:
            util_deficit = np.sum(np.abs(np.array(host_cpu_utilizations) - self.target_util))
            phi_util = 1.0 - (1.0 / n_hosts) * util_deficit
        else:
            phi_util = 0.0

        # Composite calculation
        # w[0] -> SLA/performance, w[1] -> energy, w[2] -> utilizations/cost
        r_t = (self.w[0] * phi_sla) + (self.w[1] * phi_energy) + (self.w[2] * phi_util)

        return float(r_t)

    def update_weights(self, delta_perf: float, delta_energy: float, delta_cost: float) -> None:
        """
        Applies end-of-episode adaptive weight update rule:
        w_i(k+1) = w_i(k) + alpha_w * delta_util_i(k)
        Followed by constrained projection to satisfy floors and sum-to-one bounds.
        """
        updates = np.array([delta_perf, delta_energy, delta_cost], dtype=np.float32)

        # Apply learning update
        self.w += self.alpha_w * updates

        # Constrained Projection to respect weight floors
        # Subtract floors to evaluate residual weight pools
        residuals = np.array([
            self.w[0] - self.w_sla_floor,
            self.w[1] - self.w_energy_floor,
            self.w[2] - self.w_cost_floor
        ], dtype=np.float32)

        # Clip residuals to be non-negative
        residuals = np.clip(residuals, a_min=0.0, a_max=None)
        # w[0] -> SLA/performance, w[1] -> energy, w[2] -> utilization/cost
        # Normalize the residuals to fill up the remaining pool (1.0 - Sum of floors = 0.6)
        residual_sum = np.sum(residuals)
        target_residual_sum = 1.0 - self.sum_floors # 0.6

        if residual_sum > 0:
            residuals = (residuals / residual_sum) * target_residual_sum
        else:
            # Fallback to uniform distribution of residual pool if everything collapsed
            residuals = np.ones_like(residuals) * (target_residual_sum / 3.0)

        # Re-add the floors to obtain normalized, bounded weights
        self.w[0] = residuals[0] + self.w_sla_floor
        self.w[1] = residuals[1] + self.w_energy_floor
        self.w[2] = residuals[2] + self.w_cost_floor

        self.weight_history.append(self.get_current_weights())

    def get_current_weights(self) -> dict:
        return {
            "w_sla": float(self.w[0]),
            "w_energy": float(self.w[1]),
            "w_cost": float(self.w[2])
        }
