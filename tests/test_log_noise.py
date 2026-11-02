"""FIX 7 (incident): truncation warnings must not drown the real errors.

21,062 "Episode truncated" / "Eval episode truncated" warnings made up 44% of
the log volume of the incident. They are routine: the SIM-10 episode cap
truncates most episodes by design. Per-episode detail moves to DEBUG and a
single INFO summary is emitted every 50 episodes.
"""

import logging
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql import train
from d2ql.env import OBS_DIM


class TruncatingEnv:
    """Every episode reports as truncated (the SIM-10 cap), as in production."""

    def __init__(self, finished: int = 38, n_cloudlets: int = 50):
        self.finished = finished
        self.n_cloudlets = n_cloudlets
        self.episodes = 0

    def reset(self, cloudlets=None):
        self.episodes += 1
        return np.zeros(OBS_DIM, dtype=np.float32), {"n_cloudlets": len(cloudlets or [])}

    def step(self, action):
        info = {
            "makespan": 120.0,
            "energy": 1.0,
            "cost": 0.1,
            "finished_cloudlets": self.finished,
            "n_cloudlets": self.n_cloudlets,
            "simulation_running": False,
            "episode_complete": False,
        }
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        return obs, -0.5, True, True, info  # terminated and truncated


class GreedyAgent:
    def select_action(self, obs, evaluate=False):
        return 0


def _truncation_records(caplog, level: int | None = None):
    return [
        r
        for r in caplog.records
        if "truncated" in r.getMessage() and (level is None or r.levelno == level)
    ]


# ---------------------------------------------------------------------------
# Per-episode truncation -> DEBUG
# ---------------------------------------------------------------------------


def test_training_episode_truncation_is_debug(caplog):
    caplog.set_level(logging.DEBUG)
    info = {
        "finished_cloudlets": 38,
        "n_cloudlets": 50,
        "simulation_running": False,
    }

    train._log_truncation("Episode 7", info, 1300)

    records = _truncation_records(caplog, logging.DEBUG)
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    # The message keeps the original wording, so grep patterns still work.
    assert "Episode 7 truncated: 38/50 cloudlets finished" in records[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_eval_truncation_is_debug(caplog):
    """A short held-out eval run: the real ``_run_heldout_eval`` code path."""
    caplog.set_level(logging.DEBUG)
    env = TruncatingEnv()

    metrics = train._run_heldout_eval(env, GreedyAgent(), [[object()] for _ in range(3)])

    assert metrics["eval_truncated_episodes"] == 3
    records = _truncation_records(caplog, logging.DEBUG)
    assert len(records) == 3, "one DEBUG line per truncated eval episode"
    assert all("Eval episode truncated" in r.getMessage() for r in records)
    # ... plus a single INFO summary for the window, and nothing above INFO.
    summaries = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("last 3 episodes: 3 truncated" in r.getMessage() for r in summaries)
    # Nothing at WARNING or worse: routine truncation is not an incident.
    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "truncat" in r.getMessage().lower()
    ]


# ---------------------------------------------------------------------------
# One INFO summary every 50 episodes
# ---------------------------------------------------------------------------


def test_summary_is_logged_every_50_episodes(caplog):
    caplog.set_level(logging.DEBUG)
    summary = train.TruncationSummary("training")

    for episode in range(1, 101):
        truncated = episode % 2 == 0
        info = {
            "finished_cloudlets": 38 if truncated else 50,
            "n_cloudlets": 50,
            "simulation_running": False,
        }
        if truncated:
            train._log_truncation(f"Episode {episode}", info, 300)
        summary.add(info, truncated)

    summaries = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(summaries) == 2, [r.getMessage() for r in summaries]
    for record in summaries:
        message = record.getMessage()
        assert "last 50 episodes: 25 truncated" in message
        assert "mean 44.0/50 cloudlets finished" in message

    # 50 DEBUG lines for the truncated episodes, and no warnings at all.
    assert len(_truncation_records(caplog, logging.DEBUG)) == 50
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_summary_flush_reports_a_partial_window(caplog):
    caplog.set_level(logging.DEBUG)
    summary = train.TruncationSummary("training")
    for _ in range(3):
        summary.add({"finished_cloudlets": 40, "n_cloudlets": 50}, True)

    summary.flush()
    summary.flush()  # idempotent: nothing left to report

    summaries = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.INFO and "last 3 episodes" in r.getMessage()
    ]
    assert len(summaries) == 1
    assert "3 truncated" in summaries[0]
    assert "mean 40.0/50 cloudlets finished" in summaries[0]


def test_summary_respects_the_configured_window(caplog):
    caplog.set_level(logging.INFO)
    summary = train.TruncationSummary("eval", every=5)
    for _ in range(10):
        summary.add({"finished_cloudlets": 10, "n_cloudlets": 20}, False)

    summaries = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(summaries) == 2
    assert all("last 5 episodes: 0 truncated" in r.getMessage() for r in summaries)


def test_no_per_episode_truncation_warning_remains():
    """Guard against a regression that re-introduces per-episode WARNINGs."""
    source = open(train.__file__).read()
    for call in re.finditer(r"logger\.warning\((?:[^)]*)\)", source):
        assert "truncated" not in call.group(0), (
            f"per-episode truncation still logged as a warning: {call.group(0)[:80]}"
        )
