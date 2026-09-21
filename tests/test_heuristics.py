import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from d2ql.heuristics import (  # noqa: E402
    HEURISTICS,
    available_heuristics,
    build_heuristic,
)


def _obs(cpu, ram, queue=0.5):
    """Build an observation in the env's layout: [cpu x N, ram x N, queue]."""
    cpu = np.asarray(cpu, dtype=np.float32)
    ram = np.asarray(ram, dtype=np.float32)
    return np.concatenate([cpu, ram, np.array([queue], dtype=np.float32)])


def test_registry_has_four_heuristics():
    assert set(available_heuristics()) == {
        "round_robin",
        "random",
        "least_loaded",
        "least_combined",
    }
    assert len(HEURISTICS) == 4


def test_unknown_heuristic_raises():
    with pytest.raises(ValueError):
        build_heuristic("does_not_exist", n_hosts=4)


def test_round_robin_cycles_hosts():
    p = build_heuristic("round_robin", n_hosts=4)
    p.reset()
    obs = _obs([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    actions = [p.select_action(obs) for _ in range(8)]
    assert actions == [0, 1, 2, 3, 0, 1, 2, 3]


def test_round_robin_resets_each_episode():
    p = build_heuristic("round_robin", n_hosts=4)
    obs = _obs([0.0] * 4, [0.0] * 4)
    p.select_action(obs)
    p.select_action(obs)
    p.reset()
    assert p.select_action(obs) == 0


def test_least_loaded_picks_min_cpu():
    p = build_heuristic("least_loaded", n_hosts=4)
    obs = _obs([0.9, 0.1, 0.5, 0.7], [0.0, 0.0, 0.0, 0.0])
    assert p.select_action(obs) == 1


def test_least_combined_uses_ram_too():
    p = build_heuristic("least_combined", n_hosts=4)
    # host 0 has the lowest CPU but high RAM; host 3 wins on cpu+ram.
    obs = _obs([0.1, 0.5, 0.5, 0.2], [0.9, 0.1, 0.5, 0.1])
    assert p.select_action(obs) == 3


def test_random_is_bounded_and_deterministic_given_seed():
    obs = _obs([0.0] * 4, [0.0] * 4)
    a = build_heuristic("random", n_hosts=4, seed=123)
    a.reset()
    seq_a = [a.select_action(obs) for _ in range(20)]
    assert all(0 <= x < 4 for x in seq_a)

    b = build_heuristic("random", n_hosts=4, seed=123)
    b.reset()
    seq_b = [b.select_action(obs) for _ in range(20)]
    assert seq_a == seq_b


def test_all_heuristics_return_valid_host_index():
    obs = _obs([0.2, 0.4, 0.6, 0.8], [0.5, 0.5, 0.5, 0.5])
    for name in available_heuristics():
        p = build_heuristic(name, n_hosts=4, seed=7)
        p.reset()
        for _ in range(10):
            action = p.select_action(obs)
            assert 0 <= action < 4
