"""Single source of truth for the host power model.

The Java gateway owns the simulation, but the Python reward needs ``E_step_max``
(the energy consumed if every host ran at ``P_max`` for one simulated second).
Those constants used to be hard-coded in ``reward.py`` while the gateway kept its
own copy, so changing one side silently rescaled the reward. They live here and
``tests/test_power.py`` parses ``SimulationGateway.java`` to assert both sides
still agree.

Power model (convex, heterogeneous), per host ``i``:

    P_i(u) = P_idle_i + (P_max_i - P_idle_i) * u ** gamma_i

with ``u`` the CPU utilisation in [0, 1].
"""

from __future__ import annotations

# One entry per simulated host. Keep in sync with SimulationGateway.java.
HOST_IDLE_WATT = (100.0, 80.0, 120.0, 90.0)
HOST_MAX_WATT = (250.0, 200.0, 300.0, 220.0)
HOST_GAMMA = (1.8, 1.5, 2.0, 1.6)
# USD per watt-hour, differentiated per host so cost is not a constant multiple
# of energy.
HOST_PRICE_PER_WATT = (0.048, 0.024, 0.048, 0.030)

#: Simulated seconds per gateway step (``SimulationGateway.step`` -> runFor(1.0)).
STEP_SECONDS = 1.0


def e_step_max_wh(step_seconds: float = STEP_SECONDS) -> float:
    """Energy (Wh) of one step with every host at ``P_max``: ~0.2694 Wh."""
    return sum(HOST_MAX_WATT) * float(step_seconds) / 3600.0


def idle_wh_per_step(step_seconds: float = STEP_SECONDS) -> float:
    """Idle floor (Wh) of one step: what the datacenter burns with no load.

    ``sum(HOST_IDLE_WATT) = 390 W`` -> 0.1083 Wh per simulated second. Since the
    reward is energy-only, this constant dominates the episode return (~86 % of
    it), and subtracting it is what isolates the part the scheduler controls.
    """
    return sum(HOST_IDLE_WATT) * float(step_seconds) / 3600.0
