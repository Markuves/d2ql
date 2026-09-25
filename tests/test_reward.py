import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from d2ql.reward import RewardManager

CONFIG = {
    "reward": {
        "target_utilization": 0.70,
    }
}


@pytest.fixture
def reward_manager():
    return RewardManager(CONFIG)


def test_energy_only_bounds_negative_one(reward_manager):
    # At max energy (all 4 hosts at max watt for 1s), delta = E_step_max -> r = -1
    r_max = reward_manager.compute_step_reward(energy_delta=reward_manager.e_step_max)
    assert r_max == pytest.approx(-1.0, abs=1e-6)


def test_energy_only_bounds_zero_at_zero_delta(reward_manager):
    r_zero = reward_manager.compute_step_reward(energy_delta=0.0)
    assert r_zero == pytest.approx(0.0, abs=1e-6)


def test_energy_only_linear_in_delta(reward_manager):
    # r_t is linear in energy_delta (not cumulative)
    r_small = reward_manager.compute_step_reward(energy_delta=reward_manager.e_step_max * 0.5)
    r_large = reward_manager.compute_step_reward(energy_delta=reward_manager.e_step_max)
    assert r_small == pytest.approx(-0.5, abs=1e-6)
    assert r_large == pytest.approx(-1.0, abs=1e-6)


def test_energy_only_ignores_extra_args(reward_manager):
    # API compatibility: unused SLA/util args are accepted and ignored
    r = reward_manager.compute_step_reward(
        energy_delta=0.1,
        sla_violations_this_step=5.0,
        host_cpu_utilizations=[0.9, 0.9, 0.9, 0.9],
    )
    expected = -0.1 / reward_manager.e_step_max
    assert r == pytest.approx(expected, abs=1e-6)


def test_update_weights_no_op(reward_manager):
    reward_manager.update_weights(delta_perf=0.1, delta_energy=-0.1, delta_cost=0.0)
    w = reward_manager.get_current_weights()
    assert w["w_energy"] == 1.0
    assert w["w_sla"] == 0.0
    assert w["w_cost"] == 0.0
