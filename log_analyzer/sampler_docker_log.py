import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# --- Event detection patterns -------------------------------------------------

# Single restart marker. The CUDA banner and this line both fire per
# container start; counting both would double the restart count.
RESTART_MARKER = "Loaded config from"

# Crash markers. A single crash typically matches several of these across
# its traceback lines; clustering (below) collapses them into one event.
CRASH_MARKERS = [
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"socket\.gaierror"),
    re.compile(r"ConnectionRefusedError"),
    re.compile(r"IndexError: pop from an empty deque"),
    re.compile(r"py4j\.Py4J\w*Error"),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"^\[ERROR\]"),
]

WARNING_MARKER = re.compile(r"\[WARNING\]")

# Crash markers closer than this many lines apart are treated as one event.
CRASH_CLUSTER_GAP = 5

# Window sizes (lines).
RESTART_CONTEXT_BEFORE = 5
RESTART_CONTEXT_AFTER = 10
CRASH_CONTEXT_BEFORE = 5
CRASH_CONTEXT_AFTER = 60   # generous: captures full chained tracebacks

# Sample sizes.
WARNING_SAMPLE_SIZE = 200
BACKGROUND_SAMPLE_SIZE = 1000


# --- Line parsing ---------------------------------------------------------------

def is_json_log_line(line: str) -> bool:
    """Cheap structural check before paying for json.loads."""
    return line.startswith('{"log"')


def extract_message(line: str) -> str | None:
    """Pull the 'log' field out of a Docker json-file driver line."""
    try:
        return json.loads(line)["log"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# --- Event detection --------------------------------------------------------------

def cluster_indices(indices: list[int], gap: int) -> list[int]:
    """
    Collapse nearby line numbers into cluster representatives
    (the first line of each cluster). Consecutive crash markers within
    `gap` lines belong to the same traceback event.
    """
    if not indices:
        return []
    clusters = [indices[0]]
    for i in indices[1:]:
        if i - clusters[-1] > gap:
            clusters.append(i)
    return clusters


def first_pass_find_events(path: Path) -> dict[str, list[int]]:
    """
    Scan the file once and record 0-based line numbers for each event type.
    Crash hits are clustered so one crash counts once.
    """
    events = {"restart": [], "crash_raw": [], "warning": []}
    with path.open("r", errors="replace") as f:
        for i, line in enumerate(f):
            if not is_json_log_line(line):
                continue
            msg = extract_message(line)
            if msg is None:
                continue
            if RESTART_MARKER in msg:
                events["restart"].append(i)
            if any(p.search(msg) for p in CRASH_MARKERS):
                events["crash_raw"].append(i)
            if WARNING_MARKER.search(msg):
                events["warning"].append(i)
    events["crash"] = cluster_indices(events["crash_raw"], CRASH_CLUSTER_GAP)
    return events


def build_event_windows(events: dict[str, list[int]]) -> set[int]:
    """
    Expand restart and crash line numbers into context windows.
    Warnings are excluded here; they are subsampled separately.
    Overlapping windows merge naturally since indices live in a set.
    """
    keep: set[int] = set()
    for i in events["restart"]:
        for j in range(max(0, i - RESTART_CONTEXT_BEFORE),
                       i + RESTART_CONTEXT_AFTER + 1):
            keep.add(j)
    for i in events["crash"]:
        for j in range(max(0, i - CRASH_CONTEXT_BEFORE),
                       i + CRASH_CONTEXT_AFTER + 1):
            keep.add(j)
    return keep


# --- Sampling ---------------------------------------------------------------------

def reservoir_sample(path: Path, k: int, exclude: set[int],
                     rng: random.Random) -> dict[int, str]:
    """
    Uniform sample of k lines from the file, skipping indices in `exclude`.
    Algorithm R: fill the first k slots, then for the n-th seen line replace
    slot j with probability k/n. The reservoir is a slot-indexed list so
    eviction is a uniform slot choice. O(file size) time, O(k) memory.
    Returns {line_number: raw_line}.
    """
    slots: list[tuple[int, str] | None] = [None] * k
    n_seen = 0
    with path.open("r", errors="replace") as f:
        for i, line in enumerate(f):
            if i in exclude:
                continue
            if n_seen < k:
                slots[n_seen] = (i, line)
            else:
                j = rng.randint(0, n_seen)
                if j < k:
                    slots[j] = (i, line)
            n_seen += 1
    return {i: line for slot in slots if slot is not None
            for i, line in [slot]}


def sample_indices(indices: list[int], k: int,
                   rng: random.Random) -> set[int]:
    """Uniform sample of k indices from an in-memory list."""
    if len(indices) <= k:
        return set(indices)
    return set(rng.sample(indices, k))


def collect_lines(path: Path, wanted: set[int]) -> dict[int, str]:
    """Collect raw lines for a set of 0-based line numbers."""
    out: dict[int, str] = {}
    with path.open("r", errors="replace") as f:
        for i, line in enumerate(f):
            if i in wanted:
                out[i] = line
    return out


# --- Output -------------------------------------------------------------------------

def write_lines(path: Path, indexed_lines: dict[int, str]):
    """Write sampled lines sorted by original line number."""
    with path.open("w") as f:
        for i in sorted(indexed_lines):
            f.write(indexed_lines[i])


def summarize(indexed_lines: dict[int, str]) -> dict:
    """Log-level distribution over a sample, for sanity checking."""
    levels = Counter()
    for raw in indexed_lines.values():
        msg = extract_message(raw)
        if msg is None:
            levels["non_json"] += 1
            continue
        m = re.search(r"\[(INFO|WARNING|ERROR|DEBUG|CRITICAL)\]", msg)
        levels[m.group(1) if m else "unstructured"] += 1
    return {"level_distribution": dict(levels)}


# --- Main ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Stratified sampler for Docker json-file driver logs.")
    ap.add_argument("log_file", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("samples"))
    ap.add_argument("--background-size", type=int,
                    default=BACKGROUND_SAMPLE_SIZE,
                    help=f"Uniform background sample size "
                         f"(default: {BACKGROUND_SAMPLE_SIZE})")
    ap.add_argument("--warning-size", type=int, default=WARNING_SAMPLE_SIZE,
                    help=f"Uniform warning sample size "
                         f"(default: {WARNING_SAMPLE_SIZE})")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Pass 1: scanning {args.log_file} for events...")
    events = first_pass_find_events(args.log_file)
    print(f"  restarts:        {len(events['restart'])}")
    print(f"  crash hits raw:  {len(events['crash_raw'])}")
    print(f"  crash clusters:  {len(events['crash'])}")
    print(f"  warnings:        {len(events['warning'])}")

    window_indices = build_event_windows(events)
    print(f"Pass 2: collecting {len(window_indices)} event-window lines...")
    event_lines = collect_lines(args.log_file, window_indices)

    warning_pick = sample_indices(events["warning"], args.warning_size, rng)
    print(f"Pass 3: collecting {len(warning_pick)} sampled warning lines...")
    warning_lines = collect_lines(args.log_file, warning_pick)

    print(f"Pass 4: reservoir sampling {args.background_size} background lines...")
    background = reservoir_sample(args.log_file, args.background_size,
                                  exclude=window_indices | warning_pick,
                                  rng=rng)

    events_path = args.outdir / "sample_events.log"
    warnings_path = args.outdir / "sample_warnings.log"
    background_path = args.outdir / "sample_background.log"
    write_lines(events_path, event_lines)
    write_lines(warnings_path, warning_lines)
    write_lines(background_path, background)

    manifest = {
        "source": str(args.log_file),
        "seed": args.seed,
        "events_found": {
            "restarts": len(events["restart"]),
            "crash_hits_raw": len(events["crash_raw"]),
            "crash_clusters": len(events["crash"]),
            "warnings": len(events["warning"]),
        },
        "samples": {
            "event_window_lines": len(event_lines),
            "warning_lines": len(warning_lines),
            "background_lines": len(background),
        },
        "event_sample_stats": summarize(event_lines),
        "warning_sample_stats": summarize(warning_lines),
        "background_sample_stats": summarize(background),
    }
    manifest_path = args.outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nWrote:")
    print(f"  {events_path}      ({len(event_lines)} lines)")
    print(f"  {warnings_path}   ({len(warning_lines)} lines)")
    print(f"  {background_path}  ({len(background)} lines)")
    print(f"  {manifest_path}")
    print("\nBackground levels:",
          manifest["background_sample_stats"]["level_distribution"])
    print("Warning sample levels:",
          manifest["warning_sample_stats"]["level_distribution"])
    print("Event sample levels:",
          manifest["event_sample_stats"]["level_distribution"])


if __name__ == "__main__":
    main()