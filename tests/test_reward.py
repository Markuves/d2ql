import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from d2ql.reward import RewardManager

CONFIG = {
    "reward": {
        "target_utilization": 0.70,
        "migration_penalty": 0.10,
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
        energy_this_step=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
        did_migrate=False
    )
    # phi_util should be 1.0, phi_sla and phi_energy both 0.0
    assert reward == pytest.approx(reward_manager.w[2] * 1.0, abs=1e-6)


def test_reward_migration_penalty_applied(reward_manager):
    reward_no_migrate = reward_manager.compute_step_reward(
        energy_this_step=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
        did_migrate=False
    )
    reward_migrate = reward_manager.compute_step_reward(
        energy_this_step=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
        did_migrate=True
    )
    assert reward_no_migrate - reward_migrate == pytest.approx(0.10, abs=1e-6)


def test_reward_sla_violation_negative(reward_manager):
    reward = reward_manager.compute_step_reward(
        energy_this_step=0.0,
        sla_violations_this_step=1.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
        did_migrate=False
    )
    # phi_sla is negative when violations > 0
    assert reward < reward_manager.w[2] * 1.0


def test_reward_energy_cost_negative(reward_manager):
    reward = reward_manager.compute_step_reward(
        energy_this_step=1.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
        did_migrate=False
    )
    # phi_energy = -energy/e_ref = -1.0, so reward should decrease
    reward_no_energy = reward_manager.compute_step_reward(
        energy_this_step=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[0.70, 0.70, 0.70, 0.70],
        did_migrate=False
    )
    assert reward < reward_no_energy


def test_reward_empty_host_utilizations(reward_manager):
    # Should not crash, phi_util defaults to 0.0
    reward = reward_manager.compute_step_reward(
        energy_this_step=0.0,
        sla_violations_this_step=0.0,
        host_cpu_utilizations=[],
        did_migrate=False
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
