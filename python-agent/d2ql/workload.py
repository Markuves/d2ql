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


# Column layout of the *cleaned* trace (prep/workload_clean.csv). This variant
# ships WITH a header row and a DIFFERENT column order than the raw positional
# Azure vmtable dump, so we can't rely on positional names alone. These are the
# names produced by the prep pipeline; we map them onto the internal COLUMN_NAMES.
CLEAN_HEADER = {
    "subscriptionid", "deploymentid", "vmcreated", "vmdeleted",
    "maxcpu", "avgcpu", "p95maxcpu", "vmcategory",
    "vmcorecountbucket", "vmmemorybucket", "lifetime", "corehour",
}
# Map clean-CSV header -> internal COLUMN_NAMES entry.
CLEAN_TO_INTERNAL = {
    "vmcreated": "submitted_at",
    "vmdeleted": "deadline",
    "maxcpu": "cpu_max",
    "avgcpu": "cpu_avg",
    "p95maxcpu": "cpu_p95",
    "vmcategory": "vm_category",
    "vmcorecountbucket": "num_pes",
    "vmmemorybucket": "memory_gb",
}


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
        trace_path: str = "prep/workload_clean.csv",
        episode_length: int = 50,
        mi_scale: float = 1000.0,
        seed: Optional[int] = None,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.episode_length = episode_length
        self.mi_scale = mi_scale
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self._df: pd.DataFrame = self._load()
        self._windows: list[pd.DataFrame] = self._partition()
        logger.info(
            "AzureTraceLoader ready: %d rows -> %d episode windows of length %d.",
            len(self._df),
            len(self._windows),
            self.episode_length,
        )

    # ------------------------------------------------------------------
    # Loading and partitioning
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if not self.trace_path.exists():
            raise FileNotFoundError(
                f"Trace file not found at {self.trace_path}. "
                "Place the Azure VM trace at data/workload.csv.gz or the cleaned "
                "trace at prep/workload_clean.csv."
            )

        logger.info("Loading trace from %s ...", self.trace_path)

        # The cleaned trace (prep/workload_clean.csv) ships WITH a header row and
        # a different column order than the raw positional Azure vmtable dump. We
        # peek at the first line to decide whether headers are present so we don't
        # silently misalign every column (which would produce garbage training
        # data). The raw .csv.gz / .csv has NO header and is read positionally.
        first_line = self._peek_first_line()
        has_header = bool(first_line) and (
            first_line.split(",")[0].strip() in CLEAN_HEADER
            or any(tok in CLEAN_HEADER for tok in first_line.split(",")[:3])
        )

        opener = gzip.open if self.trace_path.suffix == ".gz" else open
        with opener(self.trace_path, "rt") as f:
            if has_header:
                # Read with the real header, then rename to the internal schema.
                df = pd.read_csv(f, low_memory=False)
                rename = {k: v for k, v in CLEAN_TO_INTERNAL.items() if k in df.columns}
                df = df.rename(columns=rename)
                # Keep only the columns the loader/agent understand; drop extras
                # (subscriptionid, deploymentid, lifetime, corehour, ...).
                keep = [c for c in COLUMN_NAMES if c in df.columns]
                df = df[keep]
            else:
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

    def _peek_first_line(self) -> str:
        """Read just the first line (handles gzip transparently) for header probing."""
        opener = gzip.open if self.trace_path.suffix == ".gz" else open
        with opener(self.trace_path, "rt") as f:
            return f.readline()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce types and derive MI.

        NOTE: no rows are dropped. The cleaned trace must keep EVERY row from
        workload_clean.csv (per explicit user request) — including rows with
        missing/identical submitted_at/deadline. Those are repaired with
        defensive clips instead of being discarded, so the full workload is
        preserved for training.
        """

        # Coerce numeric columns (strings -> NaN instead of raising)
        for col in ["submitted_at", "deadline", "cpu_avg", "cpu_max", "memory_gb"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Repair, don't drop: rows missing submitted_at/deadline get safe defaults
        df["submitted_at"] = df["submitted_at"].fillna(0.0)
        df["deadline"] = df["deadline"].fillna(0.0)

        # Ensure deadline is strictly after submission (clip, never drop)
        df.loc[df["deadline"] <= df["submitted_at"], "deadline"] = (
            df["submitted_at"] + 1.0
        )

        # Estimate million instructions from average CPU and duration
        duration = (df["deadline"] - df["submitted_at"]).clip(lower=1.0)
        df["mi"] = (df["cpu_avg"].fillna(10.0) / 100.0) * duration * self.mi_scale

        # num_pes comes directly from the core count bucket column
        df["num_pes"] = pd.to_numeric(df["num_pes"], errors="coerce").fillna(1).clip(lower=1).astype(int)

        # Memory in MB from the GB bucket column
        df["memory_mb"] = pd.to_numeric(df["memory_gb"], errors="coerce").fillna(0.5) * 1024.0

        logger.info("Cleaned trace shape: %s (no rows dropped)", df.shape)
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
        Yield n_episodes lists of CloudletSpec, sampling windows with
        replacement. shuffle=True (H3 mode) randomises window order each
        epoch so the agent sees varied workload orderings.
        """
        window_indices = list(range(len(self._windows)))
        yielded = 0

        while yielded < n_episodes:
            if shuffle:
                self.rng.shuffle(window_indices)
            for idx in window_indices:
                if yielded >= n_episodes:
                    break
                yield self._window_to_specs(self._windows[idx])
                yielded += 1

    def sample_episode(self) -> list[CloudletSpec]:
        """Return a single randomly sampled episode window."""
        window = self.rng.choice(self._windows)
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
