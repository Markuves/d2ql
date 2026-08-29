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
        mi_scale: float = 1000.0,
        seed: Optional[int] = None,
        holdout_frac: float = 0.0,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.episode_length = episode_length
        self.mi_scale = mi_scale
        self.holdout_frac = holdout_frac
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self._df: pd.DataFrame = self._load()
        self._windows: list[pd.DataFrame] = self._partition()
        self._split_holdout()
        logger.info(
            "AzureTraceLoader ready: %d rows -> %d episode windows (%d train / %d eval / length %d).",
            len(self._df),
            len(self._windows),
            len(self._train_windows),
            len(self._test_windows),
            self.episode_length,
        )

    # ------------------------------------------------------------------
    # Train / eval split (C1 fix: held-out evaluation on unseen windows)
    # ------------------------------------------------------------------

    def _split_holdout(self) -> None:
        """Hold out the trailing `holdout_frac` fraction of windows (chronological)."""
        n = len(self._windows)
        if self.holdout_frac > 0.0 and n > 1:
            n_test = max(1, int(round(n * self.holdout_frac)))
            self._train_windows = self._windows[: n - n_test]
            self._test_windows = self._windows[n - n_test :]
        else:
            self._train_windows = self._windows
            self._test_windows = self._windows
        logger.info("Holdout split: holdout_frac=%.2f -> %d train / %d eval windows",
                    self.holdout_frac, len(self._train_windows), len(self._test_windows))

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
        Yield n_episodes lists of CloudletSpec, sampling TRAIN windows with
        replacement. shuffle=True (H3 mode) randomises window order each
        epoch so the agent sees varied workload orderings.
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
        """
        Deterministically sample n_episodes windows from the HELD-OUT eval
        set (C1 fix). Used for fair, epsilon-free evaluation and business-metric
        reporting on workload the agent never trained on.
        """
        if not self._test_windows:
            return []
        indices = list(range(len(self._test_windows)))
        self.rng.shuffle(indices)
        return [
            self._window_to_specs(self._test_windows[idx])
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
