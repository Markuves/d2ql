import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from d2ql.reward import RewardManager

CONFIG = {
    "reward": {
        "target_utilization": 0.70,
        "weight_lr": 0.01,
        "w_perf_floor": 0.2,
        "w_energy_floor": 0.1,
        "w_cost_floor": 0.1,
        "w_perf_init": 0.4,
        "w_energy_init": 0.3,
        "w_cost_init": 0.3,
    }
}


@pytest.fixture
def reward_manager():
    return RewardManager(CONFIG, e_ref=1.0, v_max=1.0)


def test_initial_weights_sum_to_one(reward_manager):
    w = reward_manager.get_current_weights()
    total = w["w_sla"] + w["w_energy"] + w["w_cost"]
    assert abs(total - 1.0) < 1e-6


def test_initial_weights_match_config(reward_manager):
    w = reward_manager.get_current_weights()
    assert abs(w["w_sla"] - 0.4) < 1e-6
    assert abs(w["w_energy"] - 0.3) < 1e-6
    assert abs(w["w_cost"] - 0.3) < 1e-6


def test_reward_no_violations_full_utilization(reward_manager):
    # All hosts at target utilization, no SLA violations, no energy cost
    reward = reward_manager.compute_step_reward(
        energy_delta=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
    )
    # phi_util should be 1.0, phi_sla and phi_energy both 0.0
    assert reward == pytest.approx(reward_manager.w[2] * 1.0, abs=1e-6)


def test_reward_sla_violation_negative(reward_manager):
    reward = reward_manager.compute_step_reward(
        energy_delta=0.0,
        sla_violations_this_step=1.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
    )
    # phi_sla is negative when violations > 0
    assert reward < reward_manager.w[2] * 1.0


def test_reward_energy_delta_negative(reward_manager):
    # A1 fix: the energy term must use the per-step delta, so a positive delta
    # reduces reward the same way as the old cumulative value did assumption-wise.
    reward = reward_manager.compute_step_reward(
        energy_delta=1.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
    )
    reward_no_energy = reward_manager.compute_step_reward(
        energy_delta=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
    )
    # phi_energy = -delta/e_ref = -1.0, so reward should decrease
    assert reward < reward_no_energy


def test_reward_energy_delta_scales_linearly(reward_manager):
    # A1: reward from energy must be linear in the delta (not in the cumulative
    # magnitude), so a large delta has the same bounded effect regardless of
    # how far along the episode we are.
    r_small = reward_manager.compute_step_reward(
        energy_delta=0.5,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
    )
    r_large = reward_manager.compute_step_reward(
        energy_delta=1.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
    )
    # phi_energy doubles: -0.5 -> -1.0, so the delta between the two rewards
    # equals the single-delta penalty magnitude.
    assert (r_small - r_large) == pytest.approx(0.5 * reward_manager.w[1], abs=1e-6)


def test_reward_empty_host_utilizations(reward_manager):
    # Should not crash, phi_util defaults to 0.0
    reward = reward_manager.compute_step_reward(
        energy_delta=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[],
    )
    assert isinstance(reward, float)


def test_update_weights_stay_summed_to_one(reward_manager):
    reward_manager.update_weights(
        delta_perf=0.05,
        delta_energy=-0.02,
        delta_cost=0.01
    )
    w = reward_manager.get_current_weights()
    total = w["w_sla"] + w["w_energy"] + w["w_cost"]
    assert abs(total - 1.0) < 1e-6


def test_update_weights_respect_floors(reward_manager):
    # Drive weights hard in one direction repeatedly
    for _ in range(100):
        reward_manager.update_weights(
            delta_perf=-1.0,
            delta_energy=1.0,
            delta_cost=1.0
        )
    w = reward_manager.get_current_weights()
    assert w["w_sla"] >= 0.2
    assert w["w_energy"] >= 0.1
    assert w["w_cost"] >= 0.1


def test_weight_history_recorded(reward_manager):
    reward_manager.update_weights(0.01, 0.01, 0.01)
    reward_manager.update_weights(0.01, 0.01, 0.01)
    assert len(reward_manager.weight_history) == 2
