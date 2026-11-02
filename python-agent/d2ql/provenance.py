"""Run provenance: which code and which config produced a persisted result.

Without this, two rows sharing an ``experiment_id`` can come from different code
(that is exactly what happened between the published sweep and the current
branch: observation 12→11 dims, dispatch trace→LPT). Every row now carries the
git commit, a hash of the full run config, a UTC timestamp and the environment
shape, so a CSV/parquet can be filtered and labelled by provenance
(docs/backlog.md, CSV-04 / EVAL-06).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: Bumped whenever the persisted schema or the environment changes in a way that
#: makes rows incomparable with previous ones.
SCHEMA_VERSION = 2

_REPO_ROOT = Path(__file__).resolve().parents[2]


def git_sha() -> str:
    """Short commit of the running code (env override first, then git)."""
    from_env = os.environ.get("D2QL_GIT_SHA", "").strip()
    if from_env:
        return from_env
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _normalize_keys(value):
    """Return ``value`` with every mapping key recursively turned into a ``str``.

    ``json.dumps(..., sort_keys=True)`` sorts the keys by *comparing* them, so a
    config that mixes the integer precision key ``32`` with string keys raised
    ``TypeError: '<' not supported between instances of 'str' and 'int'``. The
    H4 sweep config does exactly that (``learning_rate_overrides`` and
    ``capacity.max_hidden_size`` keyed by precision), and because provenance is
    built after the last episode it killed every H4 run at the end. Stringifying
    the keys first also makes the hash independent of whether a key arrived as
    ``32`` or ``"32"``.
    """
    if isinstance(value, dict):
        return {str(k): _normalize_keys(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_keys(v) for v in value]
    return value


def config_sha256(config: dict) -> str:
    """Stable 16-char hash of the effective run config (override included)."""
    payload = json.dumps(
        _normalize_keys(config), sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_timestamp() -> str:
    """UTC ISO-8601 timestamp with second resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def provenance(config: dict, **extra: object) -> dict:
    """Provenance block to merge into ``RunResult.extra``."""
    block = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(),
        "config_sha256": config_sha256(config),
        "run_timestamp_utc": run_timestamp(),
    }
    block.update({k: v for k, v in extra.items() if v is not None})
    return block
