"""Tests for the simulation time scale (`simulation_time_scale` / `mi_scale`)."""

import gzip
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from d2ql.workload import AzureTraceLoader


def _write_trace(tmp_path, rows):
    path = tmp_path / "workload.csv.gz"
    with gzip.open(path, "wt") as fh:
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    return str(path)


@pytest.fixture()
def trace_path(tmp_path):
    # Positional columns: vm_id, sub, deploy, submitted_at, deadline, cpu_max,
    # cpu_avg, cpu_p95, category, num_pes, memory_gb.
    # duration = 100 s, cpu_avg = 50 -> mi = (50/100) * 100 * scale = 50 * scale.
    row = [1, 1, 1, 0, 100, 50, 50, 50, "Interactive", 1, 2]
    return _write_trace(tmp_path, [row])


def _mi(loader) -> float:
    return loader.sample_episode()[0].mi


def test_simulation_time_scale_scales_simulated_time(trace_path):
    loader = AzureTraceLoader(trace_path, episode_length=1, simulation_time_scale=0.5)
    assert loader.sim_time_scale == pytest.approx(0.5)
    assert _mi(loader) == pytest.approx(25.0)


def test_mi_scale_still_works_as_alias(trace_path):
    loader = AzureTraceLoader(trace_path, episode_length=1, mi_scale=0.25)
    assert loader.sim_time_scale == pytest.approx(0.25)
    assert _mi(loader) == pytest.approx(12.5)


def test_simulation_time_scale_takes_precedence_over_mi_scale(trace_path):
    loader = AzureTraceLoader(
        trace_path, episode_length=1, mi_scale=999.0, simulation_time_scale=0.5
    )
    assert loader.sim_time_scale == pytest.approx(0.5)
    assert _mi(loader) == pytest.approx(25.0)


def test_default_time_scale_is_the_historical_1000(trace_path):
    loader = AzureTraceLoader(trace_path, episode_length=1)
    assert loader.sim_time_scale == pytest.approx(1000.0)
    assert _mi(loader) == pytest.approx(50_000.0)
