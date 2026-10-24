"""Energy-only step reward.

    r_t = -energy_delta / E_step_max   with   E_step_max = sum(P_max_i) / 3600

so each step is bounded in [-1, 0] by construction and the episode return is the
normalised energy the datacenter consumed. ``E_step_max`` comes from
``d2ql.power`` (the single source of truth shared with the Java gateway), not
from config keys: there are no weights, floors or penalties to tune anymore.
"""

from __future__ import annotations

from d2ql.power import e_step_max_wh


class RewardManager:
    """Energy-only reward; the only knob is the normalisation constant."""

    def __init__(self, e_step_max: float | None = None) -> None:
        self.e_step_max = float(e_step_max) if e_step_max is not None else e_step_max_wh()
        if not self.e_step_max > 0.0:
            raise ValueError("e_step_max must be > 0")

    def compute_step_reward(self, energy_delta: float) -> float:
        """r_t in [-1, 0]: 0 = no energy consumed, -1 = every host at P_max."""
        return float(-energy_delta / self.e_step_max)
