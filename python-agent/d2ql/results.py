"""Persist per-run results so H4 combinations can be compared after the fact.

For every (precision, hidden_size) combination we write:

* ``<checkpoint_dir>/result.json``  -- full detail for that single run.
* ``<results_dir>/h4_results.csv``  -- one row per combination, appended as each
  run finishes, so you can diff reward/latency/capacity across the whole sweep
  without re-opening TensorBoard.

Latency is the *model inference* latency (forward pass only), measured with
``DDQNAgent.benchmark_inference`` -- independent of how fast the Java sim runs.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Guard aggregate CSV appends so parallel runs never interleave writes.
_csv_write_lock = threading.Lock()


@dataclass
class RunResult:
    experiment_id: str
    precision: str
    bits: float
    hidden_size: int
    n_hidden_layers: int
    episodes_trained: int
    stopped_early: bool
    best_episode: int
    best_mean_reward: float
    best_eval_reward: float
    best_eval_episode: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_n_samples: int
    # Batched deploy throughput (samples/sec) — the regime where low-bit helps.
    throughput_pps: float
    throughput_batch_size: int
    params: int
    packed_size_mb: float
    # B2: compute-cost normalization axes so the Pareto compares real cost.
    flops: float
    effective_capacity_bits: float
    # C2: reward normalized per step.
    avg_steps: float
    norm_reward: float
    # C1: held-out evaluation with business metrics (on unseen workload).
    eval_episodes: int
    eval_mean_reward: float
    eval_makespan: float
    eval_sla_violations: float
    eval_sla_rate: float
    wall_clock_s: float
    device: str
    seed: int
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        # Flatten the nested extra dict into JSON text to keep the CSV 2-D.
        row["extra"] = json.dumps(row.get("extra", {}))
        return row

    @staticmethod
    def csv_columns() -> list[str]:
        return [
            "experiment_id",
            "precision",
            "bits",
            "hidden_size",
            "n_hidden_layers",
            "episodes_trained",
            "stopped_early",
            "best_episode",
            "best_mean_reward",
            "best_eval_reward",
            "best_eval_episode",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_n_samples",
            "throughput_pps",
            "throughput_batch_size",
            "params",
            "packed_size_mb",
            "flops",
            "effective_capacity_bits",
            "avg_steps",
            "norm_reward",
            "eval_episodes",
            "eval_mean_reward",
            "eval_makespan",
            "eval_sla_violations",
            "eval_sla_rate",
            "wall_clock_s",
            "device",
            "seed",
            "notes",
            "extra",
        ]


def _csv_path(results_dir: Path) -> Path:
    return results_dir / "h4_results.csv"


def save_run_result(result: RunResult, checkpoint_dir: str, results_dir: str) -> None:
    """Write result.json next to the checkpoint and append a row to results.csv."""
    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-run JSON detail.
    result_path = ckpt / "result.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    logger.info("Run result written to %s", result_path)

    # Append to the aggregate CSV. Guard with a lock so that parallel runs
    # (if ever enabled) never interleave writes.
    row = result.to_csv_row()
    columns = RunResult.csv_columns()
    csv_file = _csv_path(out_dir)
    write_header = not csv_file.exists()
    with _csv_write_lock:
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in columns})
    logger.info("Aggregate results updated: %s", csv_file)


def load_results_csv(results_dir: str) -> list[dict]:
    """Read the aggregate results CSV back into a list of row dicts."""
    csv_file = _csv_path(Path(results_dir))
    if not csv_file.exists():
        return []
    with open(csv_file, "r", newline="") as f:
        return list(csv.DictReader(f))


def results_csv_path(results_dir: str) -> Path:
    return _csv_path(Path(results_dir))
