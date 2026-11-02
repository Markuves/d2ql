"""FIX 2 (incident): one failing H4 cell must not kill the whole 66-cell sweep.

Before this fix a single exception raised inside ``_run_one_training`` escaped
``run_training`` and terminated the process: the deterministic ``config_sha256``
``TypeError`` (FIX 1) took down all 66 cells, which is why the python-agent
container restarted 50 times without ever finishing the sweep. A cell failure
must be logged with its identity and full traceback, and the sweep must continue.
"""

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql import train


def _cells(behaviour: dict):
    """Build ``(cell_id, run_config, thunk)`` triples that record what ran."""
    ran: list[str] = []

    def make(cell_id: str):
        def thunk() -> None:
            action = behaviour.get(cell_id, "ok")
            if action == "ok":
                ran.append(cell_id)
                return
            raise action
        return thunk

    cells = [(cid, {}, make(cid)) for cid in behaviour]
    return cells, ran


def test_a_failing_cell_does_not_kill_the_sweep(caplog):
    caplog.set_level(logging.DEBUG)
    cells, ran = _cells(
        {
            "cell1": "ok",
            "cell2": TypeError("'<' not supported between instances of 'str' and 'int'"),
            "cell3": "ok",
        }
    )

    summary = train._run_sweep(cells, sweep_name="H4")

    # Cells 1 and 3 ran; the sweep did not raise.
    assert ran == ["cell1", "cell3"]
    assert summary["completed"] == ["cell1", "cell3"]
    assert summary["failed"] == ["cell2"]


def test_the_failure_is_logged_with_cell_identity_and_traceback(caplog):
    caplog.set_level(logging.DEBUG)
    cells, _ = _cells({"cell1": "ok", "cell2": RuntimeError("boom"), "cell3": "ok"})

    train._run_sweep(cells, sweep_name="H4")

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    failure_logs = [r.getMessage() for r in errors if "cell2" in r.getMessage()]
    assert failure_logs, "the failing cell was not named in an ERROR log"
    joined = "\n".join(failure_logs)
    assert "Traceback" in joined, "the full traceback was not logged"
    assert "RuntimeError: boom" in joined


def test_the_sweep_logs_a_completed_vs_failed_summary(caplog):
    caplog.set_level(logging.DEBUG)
    cells, _ = _cells({"cell1": "ok", "cell2": ArithmeticError("nope"), "cell3": "ok"})

    summary = train._run_sweep(cells, sweep_name="H4")

    summaries = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "summary" in r.getMessage().lower()
    ]
    assert summaries, "no sweep summary was logged"
    line = summaries[-1]
    assert "2 completed" in line
    assert "1 failed" in line
    assert "cell2" in line  # the summary names what failed
    assert summary["n_completed"] == 2
    assert summary["n_failed"] == 1


def test_a_missing_workload_aborts_the_sweep():
    cells, ran = _cells(
        {
            "cell1": "ok",
            "cell2": FileNotFoundError("Trace file not found at data/workload.csv.gz"),
            "cell3": "ok",
        }
    )
    # Every remaining cell would fail the same way: re-raise instead of grinding.
    with pytest.raises(FileNotFoundError):
        train._run_sweep(cells, sweep_name="H4")
    assert ran == ["cell1"]


def test_keyboard_interrupt_is_not_swallowed():
    def stop() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train._run_sweep([("cell1", {}, stop), ("cell2", {}, lambda: None)], "H4")


def test_run_training_h4_loop_isolates_each_cell(monkeypatch, tmp_path):
    """The real H4 loop must call every cell even when one of them explodes."""
    calls: list[str] = []

    def fake_run_one_training(config: dict) -> None:
        exp_id = config["experiment"]["id"]
        calls.append(exp_id)
        if "h128" in exp_id:
            raise TypeError("'<' not supported between instances of 'str' and 'int'")

    monkeypatch.setattr(train, "_run_one_training", fake_run_one_training)

    config = {
        "experiment": {"id": "h4", "hypothesis": "H4", "seed": 42},
        "agent": {"precision": "32", "hidden_size": 64, "n_hidden_layers": 1},
        "training": {
            "n_episodes": 2,
            "eval_every_n_episodes": 50,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "tensorboard_dir": str(tmp_path / "tb"),
        },
        "results": {"csv_dir": str(tmp_path / "results")},
        "native_precision": {
            "enabled": True,
            "precisions": [{"precision": 32, "device": "cpu"}],
            "capacity": {"hidden_sizes": [64, 128, 256], "n_hidden_layers": [1]},
        },
    }

    train.run_training(config)  # must not raise

    assert len(calls) == 3
    assert calls == [
        "h4_32_cpu_h64_l1_s42",
        "h4_32_cpu_h128_l1_s42",
        "h4_32_cpu_h256_l1_s42",
    ]


def test_run_sweep_runs_every_cell_when_nothing_fails():
    cells, ran = _cells({"cell1": "ok", "cell2": "ok"})
    summary = train._run_sweep(cells, sweep_name="H4")
    assert ran == ["cell1", "cell2"]
    assert summary["failed"] == []
    assert summary["n_completed"] == 2


def test_sweep_cells_have_expected_artifact_helper():
    """The artifact helper is shared with the resume path (FIX 3)."""
    cfg = {"training": {"checkpoint_dir": str(Path("outputs") / "checkpoints" / "cell")}}
    assert train._cell_artifact_path(cfg) == Path("outputs/checkpoints/cell/result.json")
