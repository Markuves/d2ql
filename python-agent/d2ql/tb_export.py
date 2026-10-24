"""Export TensorBoard event files to tidy CSV time series.

TensorBoard is great for looking at a run and terrible for re-analysing 132 of
them: the data lives in protobuf event files, one directory per run. This module
converts each run directory into plain CSV, with these guarantees:

* **No downsampling.** ``EventAccumulator`` is configured with
  ``size_guidance={"scalars": 0}``; every scalar event is exported.
* **No precision loss *beyond TensorBoard's own*.** Values are written with
  ``repr(float)`` (round-trip exact) of what the event file holds. Note that
  ``SummaryWriter`` stores scalars as **float32**, so the event file already
  truncated them; the authoritative float64 numbers live in ``result.json`` /
  ``h4_results.csv``, which are copied next to the series when available.
* **Nothing is masked.** ``nan``/``inf`` are exported verbatim; a missing metric
  is never silently replaced by 0.
* **Deterministic order.** Long format sorted by ``(tag, step)``; wide files
  sorted by ``step``.
* **One file per namespace.** ``episode/*``, ``train/*``, ``eval/*``… have
  different x-axes (episode index vs global update vs single end-of-run point),
  so they are never pivoted into the same wide table.
* **Idempotent.** Re-running the export overwrites the output directory.

Output layout (``out_dir``)::

    series_long.csv    tag, step, wall_time_unix, wall_time_utc, value
    <namespace>.csv    step + one column per tag suffix (episode, train, ...)
    text.csv           tag, step, wall_time_utc, text      (only if present)
    run.json           counts, step ranges, non-finite counts, source dirs

Usage::

    python -m d2ql.tb_export --logdir outputs/tensorboard/tb_audit/20261024-064629 \
        --outdir outputs/results/tb_series/tb_audit/20261024-064629
    python -m d2ql.tb_export --logdir outputs/tensorboard --outdir outputs/results/tb_series
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: Scalar namespaces exported to their own wide file, in this order.
NAMESPACES = ("episode", "train", "buffer", "stopper", "eval", "perf", "run")
EVENT_PATTERN = "events.out.tfevents.*"


def _utc(wall_time: float) -> str:
    return datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat(timespec="microseconds")


def _num(value: float) -> str:
    """Round-trip exact representation; keeps ``nan``/``inf`` visible."""
    return repr(float(value))


def _text_of(event) -> str:
    from tensorboard.util import tensor_util

    array = tensor_util.make_ndarray(event.tensor_proto)
    if array.size == 0:
        return ""
    flat = array.reshape(-1)[0]
    if isinstance(flat, bytes):
        return flat.decode("utf-8", errors="replace")
    return str(flat)


def _load(log_dir: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(
        str(log_dir),
        size_guidance={"scalars": 0, "tensors": 0, "histograms": 0, "images": 0},
    )
    accumulator.Reload()
    return accumulator


def export_run(
    log_dir: str | Path,
    out_dir: str | Path,
    extra_meta: dict | None = None,
    snapshot: str | Path | None = None,
) -> dict:
    """Convert one run directory into CSV series and return its metadata.

    ``snapshot`` is the run's ``result.json`` (float64 metrics); it is copied
    into ``out_dir`` so the series and the authoritative numbers travel together.
    """
    log_dir = Path(log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accumulator = _load(log_dir)
    tags = sorted(accumulator.Tags().get("scalars", []))
    text_tags = sorted(accumulator.Tags().get("tensors", []))

    rows: list[tuple[str, int, float, float]] = []
    per_tag: dict[str, dict] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = [int(e.step) for e in events]
        walls = [float(e.wall_time) for e in events]
        values = [float(e.value) for e in events]
        rows.extend((tag, s, w, v) for s, w, v in zip(steps, walls, values))
        per_tag[tag] = {
            "n": len(events),
            "step_min": min(steps) if steps else None,
            "step_max": max(steps) if steps else None,
            "unique_steps": len(set(steps)),
            "nonfinite": sum(1 for v in values if not math.isfinite(v)),
            "first_wall_time_utc": _utc(min(walls)) if walls else None,
            "last_wall_time_utc": _utc(max(walls)) if walls else None,
        }

    rows.sort(key=lambda r: (r[0], r[1]))

    with open(out_dir / "series_long.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tag", "step", "wall_time_unix", "wall_time_utc", "value"])
        for tag, step, wall, value in rows:
            writer.writerow([tag, step, _num(wall), _utc(wall), _num(value)])

    wide_files: dict[str, str] = {}
    known = set(NAMESPACES)
    buckets: dict[str, list[tuple[str, int, float]]] = {}
    for tag, step, _wall, value in rows:
        namespace = tag.split("/", 1)[0] if "/" in tag else "other"
        if namespace not in known:
            namespace = "other"
        buckets.setdefault(namespace, []).append((tag, step, value))

    for namespace in list(NAMESPACES) + ["other"]:
        bucket = buckets.get(namespace)
        if not bucket:
            continue
        by_step: dict[int, dict[str, float]] = {}
        for tag, step, value in bucket:
            short = tag.split("/", 1)[1] if "/" in tag else tag
            by_step.setdefault(step, {})[short] = value
        columns = sorted({name for values in by_step.values() for name in values})
        filename = f"{namespace}.csv"
        with open(out_dir / filename, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", *columns])
            for step in sorted(by_step):
                values = by_step[step]
                writer.writerow([step, *[(_num(values[name]) if name in values else "") for name in columns]])
        wide_files[namespace] = filename

    text_rows: list[tuple[str, int, float, str]] = []
    for tag in text_tags:
        try:
            for event in accumulator.Tensors(tag):
                text_rows.append((tag, int(event.step), float(event.wall_time), _text_of(event)))
        except Exception as exc:  # noqa: BLE001 - not every tensor is text
            logger.warning("Skipping non-text tensor tag %s: %s", tag, exc)
    # SummaryWriter.appends "/text_summary" to add_text tags; strip the plugin
    # suffix so the CSV keeps the tag the caller wrote.
    text_rows = [
        (tag[: -len("/text_summary")] if tag.endswith("/text_summary") else tag, step, wall, text)
        for tag, step, wall, text in text_rows
    ]
    if text_rows:
        with open(out_dir / "text.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tag", "step", "wall_time_utc", "text"])
            for tag, step, wall, text in sorted(text_rows):
                writer.writerow([tag, step, _utc(wall), text])

    meta = {
        "log_dir": str(log_dir),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scalar_events": len(rows),
        "n_tags": len(tags),
        "tags": tags,
        "per_tag": per_tag,
        "wide_files": wide_files,
        "series_long": "series_long.csv",
        "text_events": len(text_rows),
        "extra": extra_meta or {},
    }
    if snapshot is not None:
        snapshot = Path(snapshot)
        if snapshot.exists():
            import shutil

            shutil.copy2(snapshot, out_dir / "result.json")
            meta["result_json"] = "result.json"
        else:
            logger.warning("Snapshot %s does not exist; not copied", snapshot)
    with open(out_dir / "run.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    logger.info(
        "TensorBoard export: %d events / %d tags -> %s", len(rows), len(tags), out_dir
    )
    return meta


def _run_dirs(root: Path) -> list[Path]:
    """Directories containing at least one event file, sorted for determinism."""
    found = sorted({p.parent for p in root.rglob(EVENT_PATTERN)})
    return found


def export_tree(log_dir: str | Path, out_dir: str | Path) -> list[dict]:
    """Export every run under ``log_dir`` keeping the relative directory layout."""
    log_dir = Path(log_dir)
    out_dir = Path(out_dir)
    if any(log_dir.glob(EVENT_PATTERN)):
        return [export_run(log_dir, out_dir)]
    results = []
    for run_dir in _run_dirs(log_dir):
        relative = run_dir.relative_to(log_dir)
        results.append(export_run(run_dir, out_dir / relative))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export TensorBoard scalars to CSV")
    parser.add_argument("--logdir", required=True, help="Run dir or a tree of run dirs")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    exported = export_tree(args.logdir, args.outdir)
    print(f"exported {len(exported)} run(s) to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
