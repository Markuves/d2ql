import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql.power import e_step_max_wh
from d2ql.reward import RewardManager


@pytest.fixture
def reward_manager():
    return RewardManager()


def test_e_step_max_comes_from_the_power_model(reward_manager):
    # sum(250, 200, 300, 220) W * 1 s / 3600 = 0.2694... Wh
    assert reward_manager.e_step_max == pytest.approx(e_step_max_wh())
    assert reward_manager.e_step_max == pytest.approx(0.269444, abs=1e-6)


def test_energy_only_bounds_negative_one(reward_manager):
    # At max energy (all 4 hosts at max watt for 1s), delta = E_step_max -> r = -1
    r_max = reward_manager.compute_step_reward(reward_manager.e_step_max)
    assert r_max == pytest.approx(-1.0, abs=1e-6)


def test_energy_only_bounds_zero_at_zero_delta(reward_manager):
    assert reward_manager.compute_step_reward(0.0) == pytest.approx(0.0, abs=1e-6)


def test_energy_only_linear_in_delta(reward_manager):
    r_small = reward_manager.compute_step_reward(reward_manager.e_step_max * 0.5)
    r_large = reward_manager.compute_step_reward(reward_manager.e_step_max)
    assert r_small == pytest.approx(-0.5, abs=1e-6)
    assert r_large == pytest.approx(-1.0, abs=1e-6)


def test_reward_survives_a_stricter_normalisation(reward_manager):
    # The constant is injectable, so a different power model does not silently
    # rescale the reward.
    strict = RewardManager(e_step_max=1.0)
    assert strict.compute_step_reward(0.25) == pytest.approx(-0.25)


def test_invalid_normalisation_is_rejected():
    with pytest.raises(ValueError):
        RewardManager(e_step_max=0.0)
