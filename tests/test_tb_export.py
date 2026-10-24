"""TensorBoard -> CSV export: fidelity, layout and idempotency."""

import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

pytest.importorskip("torch.utils.tensorboard")
from d2ql.tb_export import export_run  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

PRECISE = 0.123456789012345


def _write_events(log_dir):
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=1, max_queue=1)
    for step in range(1, 6):
        writer.add_scalar("train/loss", 0.5 - 0.01 * step, step)
        writer.add_scalar("train/mean_q", -0.01 * step, step)
    for episode in range(1, 4):
        writer.add_scalar("episode/total_reward", -9.0 - episode * 0.1, episode)
        writer.add_scalar("episode/steps", 20 + episode, episode)
    writer.add_scalar("eval/mean_reward", float("nan"), 3)
    writer.add_scalar("perf/latency_mean_ms", PRECISE, 3)
    writer.add_text("run/es_reason", "episode budget exhausted", 3)
    writer.close()


@pytest.fixture
def exported(tmp_path):
    log_dir = tmp_path / "tb" / "run"
    log_dir.mkdir(parents=True)
    _write_events(log_dir)
    out_dir = tmp_path / "csv"
    meta = export_run(log_dir, out_dir, extra_meta={"seed": 42})
    return out_dir, meta


def _long(out_dir):
    with open(out_dir / "series_long.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_event_is_exported(exported):
    out_dir, meta = exported
    rows = _long(out_dir)
    # 5 loss + 5 mean_q + 3 reward + 3 steps + 1 eval + 1 perf
    assert len(rows) == 18
    assert meta["n_scalar_events"] == 18
    assert meta["per_tag"]["train/loss"]["n"] == 5


def test_values_keep_tensorboard_precision(exported):
    # SummaryWriter stores scalars as float32; the export must be bit-exact to
    # what the event file holds (no additional formatting loss).
    out_dir, _ = exported
    perf = [r for r in _long(out_dir) if r["tag"] == "perf/latency_mean_ms"][0]
    expected = float(np.float32(PRECISE))
    assert perf["value"] == repr(expected)
    assert float(perf["value"]) == expected


def test_non_finite_values_are_not_masked(exported):
    out_dir, meta = exported
    eval_row = [r for r in _long(out_dir) if r["tag"] == "eval/mean_reward"][0]
    assert eval_row["value"] == "nan"
    assert meta["per_tag"]["eval/mean_reward"]["nonfinite"] == 1


def test_wide_files_are_split_by_namespace(exported):
    out_dir, meta = exported
    assert meta["wide_files"]["train"] == "train.csv"
    assert meta["wide_files"]["episode"] == "episode.csv"

    with open(out_dir / "train.csv", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    assert list(train_rows[0].keys()) == ["step", "loss", "mean_q"]
    assert [r["step"] for r in train_rows] == ["1", "2", "3", "4", "5"]

    with open(out_dir / "episode.csv", newline="") as fh:
        episode_rows = list(csv.DictReader(fh))
    # A naive pivot by step would have merged both axes (1..3 vs 1..5).
    assert [r["step"] for r in episode_rows] == ["1", "2", "3"]
    assert set(episode_rows[0].keys()) == {"step", "total_reward", "steps"}


def test_text_events_are_exported(exported):
    out_dir, _ = exported
    with open(out_dir / "text.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["tag"] == "run/es_reason"
    assert rows[0]["text"] == "episode budget exhausted"


def test_metadata_describes_the_export(exported):
    out_dir, meta = exported
    on_disk = json.load(open(out_dir / "run.json"))
    assert on_disk["n_tags"] == meta["n_tags"] == 6
    assert on_disk["extra"]["seed"] == 42
    assert on_disk["series_long"] == "series_long.csv"
    assert set(on_disk["wide_files"]) == {"episode", "train", "eval", "perf"}


def test_export_is_idempotent(tmp_path):
    log_dir = tmp_path / "tb" / "run"
    log_dir.mkdir(parents=True)
    _write_events(log_dir)
    out_dir = tmp_path / "csv"

    first = export_run(log_dir, out_dir)
    second = export_run(log_dir, out_dir)

    assert first["n_scalar_events"] == second["n_scalar_events"]
    assert len(_long(out_dir)) == first["n_scalar_events"]
