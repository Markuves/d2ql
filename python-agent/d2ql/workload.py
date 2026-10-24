from __future__ import annotations

import gzip
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Azure Public Dataset V2 vmtable has no header row.
# Columns are assigned positionally.
COLUMN_NAMES = [
    "vm_id",
    "subscription_id",
    "deployment_id",
    "submitted_at",       # timestamp_vm_created, seconds from trace start
    "deadline",           # timestamp_vm_deleted, seconds from trace start
    "cpu_max",            # max CPU utilization (%)
    "cpu_avg",            # average CPU utilization (%)
    "cpu_p95",            # p95 max CPU utilization (%)
    "vm_category",        # Delay-insensitive / Interactive / Unknown
    "num_pes",            # virtual core count bucket
    "memory_gb",          # memory bucket in GB
]


@dataclass
class CloudletSpec:
    """Lightweight descriptor for a single cloudlet derived from the trace."""

    cloudlet_id: int
    mi: float           # million instructions, estimated from cpu_avg * duration
    num_pes: int        # cores required
    memory_mb: float
    submitted_at: float
    deadline: float


class AzureTraceLoader:
    """
    Loads and partitions the Azure Public Dataset V2 VM trace (vmtable).

    The trace has no header row. Columns are assigned positionally on load.
    Episodes are sampled as fixed-length windows over the chronologically
    sorted timeline, with optional random shuffling between epochs to expose
    the agent to varied workload conditions (H3).

    Usage:
        loader = AzureTraceLoader("data/workload.csv.gz", episode_length=50)
        for episode_cloudlets in loader.episodes(n_episodes=600):
            env.reset(cloudlets=episode_cloudlets)
    """

    def __init__(
        self,
        trace_path: str = "data/workload.csv.gz",
        episode_length: int = 50,
        mi_scale: Optional[float] = None,
        seed: Optional[int] = None,
        holdout_frac: float = 0.0,
        simulation_time_scale: Optional[float] = None,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.episode_length = episode_length
        # Canonical knob for how long a cloudlet occupies the simulated machine:
        # exec_time ~ mi / mips, so scaling the estimated instruction count scales
        # simulated time one-to-one. `mi_scale` is the legacy name for exactly the
        # same quantity and is only honored when `simulation_time_scale` is absent,
        # so an H4 config that sets both can never double-scale by accident.
        # 1000.0 preserves the historical default when neither key is present.
        if simulation_time_scale is not None:
            if mi_scale is not None and abs(float(mi_scale) - float(simulation_time_scale)) > 1e-12:
                logger.warning(
                    "Both mi_scale=%.6g and simulation_time_scale=%.6g are set; "
                    "using simulation_time_scale.",
                    float(mi_scale), float(simulation_time_scale),
                )
            effective_scale = float(simulation_time_scale)
        else:
            effective_scale = float(mi_scale) if mi_scale is not None else 1000.0
        self.sim_time_scale = effective_scale
        # Kept as an alias: it is the effective MI multiplier used below.
        self.mi_scale = effective_scale
        self.holdout_frac = holdout_frac
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self._df: pd.DataFrame = self._load()
        self._windows: list[pd.DataFrame] = self._partition()
        self._split_holdout()
        logger.info(
            "AzureTraceLoader ready: %d rows -> %d episode windows (%d train / %d eval / %d temporal / length %d).",
            len(self._df),
            len(self._windows),
            len(self._train_windows),
            len(self._test_windows),
            len(self._temporal_windows),
            self.episode_length,
        )

    # ------------------------------------------------------------------
    # Train / eval split (C1 fix: held-out evaluation on unseen windows)
    # ------------------------------------------------------------------

    def _split_holdout(self) -> None:
        """Split train/eval **stratified by MI**, keeping the chronological tail apart.

        A plain chronological tail is a different regime: the last 20 % of the
        trace has ~10x less work per window than the rest, so `eval_*` metrics
        measured there were not comparable with anything seen in training
        (docs/backlog.md, EVAL-01). The quality metric now uses a systematic
        sample over the MI distribution (every k-th window of the MI-sorted
        order), which is disjoint from training and has the same MI mean. The
        trailing windows are still available through ``temporal_episodes`` as a
        separate temporal-generalisation test.
        """
        n = len(self._windows)
        if self.holdout_frac > 0.0 and n > 1:
            n_test = max(1, int(round(n * self.holdout_frac)))
            # Chronological tail (temporal generalisation only).
            self._temporal_windows = self._windows[n - n_test:]

            # Systematic sample over the MI-sorted order -> matched distribution.
            mi = np.array([float(w["mi"].mean()) for w in self._windows])
            order = np.argsort(mi, kind="stable")
            step = n / float(n_test)
            # Half-step offset: take the middle of each block, not its first
            # element (otherwise the sample is systematically the lightest
            # window of every block).
            test_pos = {
                int(min(n - 1, max(0, round((k + 0.5) * step)))) for k in range(n_test)
            }
            self._test_windows = [self._windows[order[pos]] for pos in sorted(test_pos)]
            self._train_windows = [
                w for i, w in enumerate(self._windows) if i not in test_pos
            ]
        else:
            self._train_windows = self._windows
            self._test_windows = self._windows
            self._temporal_windows = self._windows

        train_mi = float(np.mean([w["mi"].mean() for w in self._train_windows])) if self._train_windows else float("nan")
        eval_mi = float(np.mean([w["mi"].mean() for w in self._test_windows])) if self._test_windows else float("nan")
        logger.info(
            "Holdout split: holdout_frac=%.2f -> %d train / %d eval windows | "
            "MI mean train=%.1f eval=%.1f | temporal tail=%d windows",
            self.holdout_frac,
            len(self._train_windows),
            len(self._test_windows),
            train_mi,
            eval_mi,
            len(self._temporal_windows),
        )

    # ------------------------------------------------------------------
    # Loading and partitioning
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if not self.trace_path.exists():
            raise FileNotFoundError(
                f"Trace file not found at {self.trace_path}. "
                "Place the Azure VM trace at data/workload.csv.gz."
            )

        logger.info("Loading trace from %s ...", self.trace_path)

        opener = gzip.open if self.trace_path.suffix == ".gz" else open
        with opener(self.trace_path, "rt") as f:
            df = pd.read_csv(
                f,
                header=None,            # file has no header row
                names=COLUMN_NAMES,     # assign names positionally
                low_memory=False,
            )

        logger.info("Raw trace shape: %s", df.shape)
        df = self._clean(df)
        df = df.sort_values("submitted_at").reset_index(drop=True)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop nulls, coerce types, clip outliers, and derive MI."""

        # Coerce numeric columns
        for col in ["submitted_at", "deadline", "cpu_avg", "cpu_max", "memory_gb"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["submitted_at", "deadline"])

        # Remove rows where deadline precedes or equals submission
        df = df[df["deadline"] > df["submitted_at"]].reset_index(drop=True)

        # Estimate million instructions from average CPU and duration
        duration = (df["deadline"] - df["submitted_at"]).clip(lower=1.0)
        df["mi"] = (df["cpu_avg"].fillna(10.0) / 100.0) * duration * self.mi_scale

        # num_pes comes directly from the core count bucket column
        df["num_pes"] = pd.to_numeric(df["num_pes"], errors="coerce").fillna(1).clip(lower=1).astype(int)

        # Memory in MB from the GB bucket column
        df["memory_mb"] = pd.to_numeric(df["memory_gb"], errors="coerce").fillna(0.5) * 1024.0

        logger.info("Cleaned trace shape: %s", df.shape)
        return df

    def _partition(self) -> list[pd.DataFrame]:
        """Slice the sorted DataFrame into non-overlapping episode windows."""
        n = len(self._df)
        windows = []
        for start in range(0, n - self.episode_length + 1, self.episode_length):
            windows.append(self._df.iloc[start : start + self.episode_length])
        if not windows:
            windows = [self._df]
        return windows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def episodes(
        self,
        n_episodes: int,
        shuffle: bool = True,
    ) -> Iterator[list[CloudletSpec]]:
        """
        Yield n_episodes lists of CloudletSpec from the TRAIN windows.

        Windows are visited with replacement *between epochs*: `shuffle=True`
        (H3 mode) reshuffles the window order each pass, so a run longer than the
        number of windows keeps sampling the whole distribution.
        """
        window_indices = list(range(len(self._train_windows)))
        yielded = 0

        while yielded < n_episodes:
            if shuffle:
                self.rng.shuffle(window_indices)
            for idx in window_indices:
                if yielded >= n_episodes:
                    break
                yield self._window_to_specs(self._train_windows[idx])
                yielded += 1

    def eval_episodes(self, n_episodes: int) -> list[list[CloudletSpec]]:
        """Sample n_episodes held-out windows **from the MI-stratified split**.

        The stratified split is disjoint from training and matches its MI
        distribution, so the reported quality is comparable with the training
        regime (EVAL-01).
        """
        if not self._test_windows:
            return []
        indices = list(range(len(self._test_windows)))
        self.rng.shuffle(indices)
        return [
            self._window_to_specs(self._test_windows[idx])
            for idx in indices[:n_episodes]
        ]

    def temporal_episodes(self, n_episodes: int) -> list[list[CloudletSpec]]:
        """Sample n_episodes windows from the **chronological tail** (last windows).

        Kept as a separate temporal-generalisation test: it answers "how does the
        policy do on the newest slice of the trace", not "how good is it on the
        workload it was trained on". Not used for the quality metric.
        """
        if not self._temporal_windows:
            return []
        indices = list(range(len(self._temporal_windows)))
        self.rng.shuffle(indices)
        return [
            self._window_to_specs(self._temporal_windows[idx])
            for idx in indices[:n_episodes]
        ]

    def sample_episode(self) -> list[CloudletSpec]:
        """Return a single randomly sampled TRAIN episode window."""
        window = self.rng.choice(self._train_windows)
        return self._window_to_specs(window)

    def _window_to_specs(self, window: pd.DataFrame) -> list[CloudletSpec]:
        specs: list[CloudletSpec] = []
        for i, (_, row) in enumerate(window.iterrows()):
            specs.append(
                CloudletSpec(
                    cloudlet_id=i,
                    mi=float(row.get("mi", self.mi_scale)),
                    num_pes=int(row.get("num_pes", 1)),
                    memory_mb=float(row.get("memory_mb", 512.0)),
                    submitted_at=float(row.get("submitted_at", 0.0)),
                    deadline=float(row.get("deadline", 3600.0)),
                )
            )
        return specs

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "n_rows": len(self._df),
            "n_windows": len(self._windows),
            "episode_length": self.episode_length,
            "mi_mean": float(self._df["mi"].mean()),
            "mi_std": float(self._df["mi"].std()),
            "num_pes_mode": int(self._df["num_pes"].mode().iloc[0]),
            "duration_mean_s": float(
                (self._df["deadline"] - self._df["submitted_at"]).mean()
            ),
        }
