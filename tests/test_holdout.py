"""EVAL-01: the hold-out used for quality must match the training distribution."""

import gzip
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql.workload import AzureTraceLoader


def _write_varied_trace(tmp_path, n_rows: int = 200):
    """Trace whose cloudlet MI varies by two orders of magnitude across the trace."""
    rows = []
    for i in range(n_rows):
        # duration fixed, cpu_avg (and therefore mi) grows with the row index
        cpu_avg = 0.1 + (i / n_rows) * 99.9
        rows.append([i, 1, 1, i, i + 100, cpu_avg, cpu_avg, cpu_avg, "Unknown", 1, 2])
    path = tmp_path / "varied.csv.gz"
    with gzip.open(path, "wt") as fh:
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    return str(path)


def _window_mi_mean(windows) -> float:
    return float(np.mean([w["mi"].mean() for w in windows]))


@pytest.fixture
def loader(tmp_path):
    return AzureTraceLoader(
        _write_varied_trace(tmp_path),
        episode_length=5,
        holdout_frac=0.2,
        simulation_time_scale=0.01,
    )


def test_holdout_is_stratified_by_mi(loader):
    train_mi = _window_mi_mean(loader._train_windows)
    eval_mi = _window_mi_mean(loader._test_windows)
    # A plain chronological tail was ~10x lighter; the systematic MI sample
    # must track the training distribution.
    assert abs(eval_mi - train_mi) / train_mi < 0.10


def test_split_sizes_add_up(loader):
    n_test = len(loader._test_windows)
    assert n_test == max(1, round(len(loader._windows) * 0.2))
    assert len(loader._train_windows) + n_test == len(loader._windows)


def test_temporal_split_keeps_the_chronological_tail(loader):
    n_test = len(loader._test_windows)
    assert loader._temporal_windows == loader._windows[-n_test:]
    assert len(loader.temporal_episodes(3)) == 3
    # The tail is a different regime by construction, which is why it is no
    # longer used for the quality metric.
    assert _window_mi_mean(loader._temporal_windows) != pytest.approx(
        _window_mi_mean(loader._train_windows)
    )


def test_eval_and_train_windows_are_disjoint(loader):
    train_ids = {id(w) for w in loader._train_windows}
    test_ids = {id(w) for w in loader._test_windows}
    assert train_ids.isdisjoint(test_ids)


def test_eval_episodes_come_from_the_stratified_split(loader):
    specs = loader.eval_episodes(2)
    assert len(specs) == 2
    assert len(specs[0]) == 5
