"""CSV persistence: upsert semantics (CSV-02) and the reward/SLA columns (CSV-01)."""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql.results import HeuristicResult, RunResult, save_heuristic_result, save_run_result


def _run_result(experiment_id: str, seed: int, reward: float) -> RunResult:
    return RunResult(
        experiment_id=experiment_id,
        precision="32",
        bits=32.0,
        hidden_size=64,
        n_hidden_layers=1,
        episodes_trained=10,
        stopped_early=False,
        best_episode=10,
        best_mean_reward=reward,
        latency_mean_ms=0.1,
        latency_p50_ms=0.1,
        latency_p95_ms=0.2,
        latency_n_samples=10,
        throughput_pps=1000.0,
        throughput_batch_size=64,
        params=1234,
        packed_size_mb=0.005,
        flops=2468.0,
        effective_capacity_bits=39488.0,
        avg_steps=50.0,
        norm_reward=reward / 50.0,
        eval_episodes=2,
        eval_mean_reward=reward,
        eval_avg_steps=50.0,
        eval_norm_reward=reward / 50.0,
        eval_makespan=1.0,
        wall_clock_s=1.0,
        device="cpu",
        seed=seed,
    )


def _rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_rerunning_a_cell_replaces_its_row(tmp_path):
    res = str(tmp_path)
    ckpt = str(tmp_path / "ckpt")
    save_run_result(_run_result("exp_32_cpu_h64_l1_s42", 42, -10.0), ckpt, res)
    save_run_result(_run_result("exp_32_cpu_h64_l1_s42", 42, -20.0), ckpt, res)

    rows = _rows(str(tmp_path / "h4_results.csv"))
    assert len(rows) == 1, "re-running the same (experiment_id, seed) must not append"
    assert float(rows[0]["best_mean_reward"]) == -20.0


def test_different_seeds_are_kept(tmp_path):
    res = str(tmp_path)
    ckpt = str(tmp_path / "ckpt")
    save_run_result(_run_result("exp_s42", 42, -10.0), ckpt, res)
    save_run_result(_run_result("exp_s41", 41, -11.0), ckpt, res)
    assert len(_rows(str(tmp_path / "h4_results.csv"))) == 2


def test_run_columns_are_reward_normalised_and_sla_free():
    cols = RunResult.csv_columns()
    assert "norm_reward" in cols and "eval_norm_reward" in cols
    assert "eval_avg_steps" in cols
    assert not any("sla" in c.lower() for c in cols)
    # The window sum used to live under best_mean_reward; the eval duplicate and
    # the SLA columns are gone (CSV-01 / SLA retirement).
    assert "best_eval_reward" not in cols
    assert "best_eval_episode" not in cols


def test_heuristic_rows_are_sla_free():
    cols = HeuristicResult.csv_columns()
    assert not any("sla" in c.lower() for c in cols)


def test_heuristic_rerun_replaces_row(tmp_path):
    res = str(tmp_path)
    ckpt = str(tmp_path / "ckpt")
    result = HeuristicResult(
        experiment_id="h5_round_robin",
        policy="round_robin",
        n_hosts=4,
        eval_episodes=2,
        eval_mean_reward=-10.0,
        eval_makespan=1.0,
        mean_energy=1.0,
        mean_cost=0.01,
        decision_latency_mean_ms=0.001,
        decision_latency_p50_ms=0.001,
        decision_latency_p95_ms=0.002,
        wall_clock_s=1.0,
        seed=42,
    )
    save_heuristic_result(result, ckpt, res)
    save_heuristic_result(result, ckpt, res)
    assert len(_rows(str(tmp_path / "h5_results.csv"))) == 1
