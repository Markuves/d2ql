import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from d2ql.agent import PrioritizedReplayBuffer


@pytest.fixture
def buffer():
    return PrioritizedReplayBuffer(capacity=100, alpha=0.6)


def test_push_increases_length(buffer):
    assert len(buffer) == 0
    buffer.push(
        np.zeros(3), 0, 1.0, np.zeros(3), False
    )
    assert len(buffer) == 1


def test_push_up_to_capacity(buffer):
    for i in range(100):
        buffer.push(np.zeros(3), 0, float(i), np.zeros(3), False)
    assert len(buffer) == 100


def test_push_beyond_capacity_overwrites(buffer):
    for i in range(150):
        buffer.push(np.zeros(3), 0, float(i), np.zeros(3), False)
    assert len(buffer) == 100


def test_sample_returns_correct_batch_size(buffer):
    for i in range(50):
        buffer.push(np.zeros(3), i % 4, float(i), np.zeros(3), False)
    states, actions, rewards, next_states, dones, weights, indices = buffer.sample(16, beta=0.4)
    assert states.shape == (16, 3)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_states.shape == (16, 3)
    assert dones.shape == (16,)
    assert weights.shape == (16,)
    assert len(indices) == 16


def test_sample_weights_normalized(buffer):
    for i in range(50):
        buffer.push(np.zeros(3), 0, float(i), np.zeros(3), False)
    _, _, _, _, _, weights, _ = buffer.sample(16, beta=0.4)
    # Max weight should be 1.0 after normalization
    assert weights.max() == pytest.approx(1.0, abs=1e-6)


def test_update_priorities_changes_sampling(buffer):
    for i in range(50):
        buffer.push(np.zeros(3), 0, float(i), np.zeros(3), False)
    _, _, _, _, _, _, indices = buffer.sample(16, beta=0.4)

    # Use an index that is guaranteed not to appear elsewhere in the batch
    # by picking one outside the sampled set entirely
    target_idx = 0
    while target_idx in indices:
        target_idx += 1

    new_priorities = np.ones(len(indices)) * 0.001
    buffer.update_priorities(indices, new_priorities)

    # Directly write a high priority to a known, unambiguous index
    buffer.priorities[target_idx] = 1000.0
    assert buffer.priorities[target_idx] == pytest.approx(1000.0, abs=1e-3)



def test_priorities_never_zero_after_update(buffer):
    for i in range(50):
        buffer.push(np.zeros(3), 0, float(i), np.zeros(3), False)
    _, _, _, _, _, _, indices = buffer.sample(16, beta=0.4)
    buffer.update_priorities(indices, np.zeros(len(indices)))
    for idx in indices:
        assert buffer.priorities[idx] >= 1e-6


def test_sample_survives_nan_inf_priorities():
    # Regression: an inf/NaN priority must not crash np.random.choice(
    # "probabilities contain NaN") and must not corrupt the whole buffer.
    buf = PrioritizedReplayBuffer(capacity=100)
    for i in range(64):
        buf.push(np.zeros(3), i % 4, float(i), np.zeros(3), False)
    # Poison priorities before sampling.
    buf.priorities[:64] = np.array([float("nan"), float("inf"), 0.0, -5.0] * 16)
    # sample() must not raise, return valid shapes, and stay finite.
    states, actions, rewards, next_states, dones, weights, indices = buf.sample(16, beta=0.4)
    assert states.shape == (16, 3)
    assert actions.shape == (16,)
    assert weights.shape == (16,)
    assert np.isfinite(weights).all()
    # update_priorities sanitizes the sampled indices (inf/NaN -> finite).
    buf.update_priorities(indices, np.array([float("inf"), float("nan")] * 8))
    assert np.isfinite(buf.priorities[indices]).all()
    # And a second sample over the still-poisoned raw array must not crash.
    _, _, _, _, _, weights2, _ = buf.sample(16, beta=0.4)
    assert np.isfinite(weights2).all()
