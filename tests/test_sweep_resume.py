"""FIX 3 (incident): ``--resume`` must skip sweep cells that already finished.

Every container restart re-ran the sweep from cell 1, because nothing looked at
the artifacts the finished cells left under ``outputs/``. With ``--resume`` a
cell whose result artifact is already complete is skipped; docker-compose passes
the flag so a restart continues where the previous container died.
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql import train


def _config(tmp_path: Path) -> dict:
    """Two-cell H4 plan: (32, cpu, h64, l1) and (32, cpu, h128, l1)."""
    return {
        "experiment": {"id": "h4", "hypothesis": "H4", "seed": 42},
        "agent": {"precision": "32", "hidden_size": 64, "n_hidden_layers": 1},
        "training": {
            "n_episodes": 2,
            "eval_every_n_episodes": 50,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "tensorboard_dir": str(tmp_path / "tb"),
        },
        "results": {"csv_dir": str(tmp_path / "results")},
        "native_precision": {
            "enabled": True,
            "precisions": [{"precision": 32, "device": "cpu"}],
            "capacity": {"hidden_sizes": [64, 128], "n_hidden_layers": [1]},
        },
    }


def _cell_ckpt(config: dict, tag: str) -> Path:
    return Path(config["training"]["checkpoint_dir"]) / tag


def _fake_training(monkeypatch, executed: list[str]) -> None:
    """Stand-in for the real training run: records the cell and writes its artifact."""

    def fake_run_one_training(config: dict) -> None:
        exp_id = config["experiment"]["id"]
        executed.append(exp_id)
        ckpt = Path(config["training"]["checkpoint_dir"])
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "result.json").write_text(
            json.dumps({"experiment_id": exp_id, "episodes_trained": 2})
        )

    monkeypatch.setattr(train, "_run_one_training", fake_run_one_training)


def _write_completed_artifact(config: dict, tag: str) -> None:
    ckpt = _cell_ckpt(config, tag)
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "result.json").write_text(json.dumps({"experiment_id": tag}))


def test_resume_skips_a_completed_cell(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    config = _config(tmp_path)
    executed: list[str] = []
    _fake_training(monkeypatch, executed)
    _write_completed_artifact(config, "32_cpu_h64_l1_s42")
    config["_resume_sweep"] = True

    train.run_training(config)

    # Cell 1 was skipped, cell 2 ran.
    assert executed == ["h4_32_cpu_h128_l1_s42"]
    skipped_logs = [
        r.getMessage()
        for r in caplog.records
        if "skipping completed cell" in r.getMessage().lower()
    ]
    assert skipped_logs, "the skipped cell was not logged"
    assert "32_cpu_h64_l1_s42" in skipped_logs[0]
    # The summary reports the skip instead of hiding it.
    summary = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "summary" in r.getMessage().lower()
    ]
    assert summary and "1 skipped" in summary[-1]


def test_without_resume_the_completed_cell_runs_again(monkeypatch, tmp_path):
    config = _config(tmp_path)
    executed: list[str] = []
    _fake_training(monkeypatch, executed)
    _write_completed_artifact(config, "32_cpu_h64_l1_s42")

    train.run_training(config)  # no --resume

    assert executed == ["h4_32_cpu_h64_l1_s42", "h4_32_cpu_h128_l1_s42"]


def test_resume_reruns_a_cell_whose_artifact_is_incomplete(monkeypatch, tmp_path):
    config = _config(tmp_path)
    executed: list[str] = []
    _fake_training(monkeypatch, executed)
    # Truncated / half-written artifact: not complete, so the cell must re-run.
    ckpt = _cell_ckpt(config, "32_cpu_h64_l1_s42")
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "result.json").write_text('{"experiment_id": "h4_32_cpu_h64')
    config["_resume_sweep"] = True

    train.run_training(config)

    assert executed == ["h4_32_cpu_h64_l1_s42", "h4_32_cpu_h128_l1_s42"]


def test_resume_reruns_a_cell_with_an_empty_artifact(monkeypatch, tmp_path):
    config = _config(tmp_path)
    executed: list[str] = []
    _fake_training(monkeypatch, executed)
    ckpt = _cell_ckpt(config, "32_cpu_h64_l1_s42")
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "result.json").write_text("")
    config["_resume_sweep"] = True

    train.run_training(config)

    assert executed == ["h4_32_cpu_h64_l1_s42", "h4_32_cpu_h128_l1_s42"]


def test_cell_is_complete_helper(tmp_path):
    config = _config(tmp_path)
    assert train._cell_is_complete(config) is False
    ckpt = Path(config["training"]["checkpoint_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "result.json").write_text(json.dumps({"experiment_id": "x"}))
    assert train._cell_is_complete(config) is True


# ---------------------------------------------------------------------------
# CLI wiring: --resume without a value, and --resume <checkpoint>
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, argv: list[str], tmp_path) -> dict:
    """Run main() with a stubbed run_training and return the config it received."""
    seen: dict = {}
    config = {
        "experiment": {"id": "h4", "hypothesis": "H4", "seed": 42},
        "agent": {"precision": "32", "hidden_size": 64, "n_hidden_layers": 1},
        "training": {
            "n_episodes": 2,
            "eval_every_n_episodes": 50,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "tensorboard_dir": str(tmp_path / "tb"),
        },
    }
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    monkeypatch.setattr(train, "load_config", lambda path: config)
    monkeypatch.setattr(train, "run_training", lambda cfg: seen.update(cfg))
    train.main()
    return seen


def test_resume_flag_enables_sweep_resume(monkeypatch, tmp_path):
    seen = _run_main(monkeypatch, ["--config", "cfg.yaml", "--resume"], tmp_path)
    assert seen.get("_resume_sweep") is True
    # A bare flag is not a checkpoint path.
    assert not seen.get("_resume_path")


def test_resume_with_a_path_also_resumes_weights(monkeypatch, tmp_path):
    ckpt = tmp_path / "checkpoint_final.pt"
    ckpt.write_bytes(b"weights")
    seen = _run_main(
        monkeypatch, ["--config", "cfg.yaml", "--resume", str(ckpt)], tmp_path
    )
    assert seen.get("_resume_sweep") is True
    assert seen.get("_resume_path") == str(ckpt)


def test_no_resume_flag_means_no_sweep_resume(monkeypatch, tmp_path):
    seen = _run_main(monkeypatch, ["--config", "cfg.yaml"], tmp_path)
    assert not seen.get("_resume_sweep")
    assert not seen.get("_resume_path")
