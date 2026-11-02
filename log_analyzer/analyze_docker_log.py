from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# --- Log grammar ---------------------------------------------------------------

RUN_START_MARKER = "Loaded config from"
RUN_CONFIG_RE = re.compile(r"Loaded config from (\S+)")
TRAINING_START_RE = re.compile(r"Starting training: (\d+) episodes")
BANNER_MARKER = "== CUDA =="

STRUCTURED_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} "
    r"\[(INFO|WARNING|ERROR|DEBUG|CRITICAL)\] "
)
TRACEBACK_START = "Traceback (most recent call last):"

# Float sub-pattern that also accepts nan and inf, so diverged training
# episodes are still parsed instead of silently dropped.
FP = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|-?inf)"

EPISODE_RE = re.compile(
    rf"Episode (\d+)/(\d+) \| Reward: {FP} \| Steps: (\d+) "
    rf"\| Updates: (\d+) \| Avg Loss: {FP} \| Mean Q: {FP} "
    rf"\| Grad: {FP} \| Epsilon: {FP} \| Cloudlets: (\d+)"
)
TRUNC_RE = re.compile(r"(?:Eval episode|Episode) (\d+) truncated: (\d+)/(\d+) cloudlets")

# First match wins. Patterns target the LAST line of an assembled traceback,
# so chained tracebacks report the root exception rather than the precursor.
EXCEPTION_PATTERNS = [
    ("socket.gaierror (java-sim name not resolvable)", r"socket\.gaierror"),
    ("ConnectionRefusedError (java-sim not listening)", r"ConnectionRefusedError"),
    ("Py4JNetworkError: empty answer (JVM died mid-call)", r"Answer from Java side is empty"),
    ("Py4JNetworkError (other)", r"Py4JNetworkError"),
    ("IndexError (py4j connection pool drained)", r"IndexError"),
    ("OutOfMemoryError", r"OutOfMemoryError"),
    ("FileNotFoundError", r"FileNotFoundError"),
    ("TimeoutError", r"TimeoutError"),
]

COMPLETION_HINTS = ("early stopping", "training complete", "finished all", "sweep complete")


# --- Data model ------------------------------------------------------------------

@dataclass
class Entry:
    ts: datetime
    stream: str
    msg: str  # trailing newline stripped, indentation preserved


@dataclass
class Crash:
    ts: datetime
    signature: str
    lines: list[str] = field(default_factory=list)
    fatal: bool = False
    episodes_before: int = 0


@dataclass
class Run:
    idx: int
    start_ts: datetime
    config: str = ""
    partial: bool = False
    planned_episodes: int | None = None
    training_start_ts: datetime | None = None
    end_ts: datetime | None = None
    episodes: list[tuple[int, float, float, datetime]] = field(default_factory=list)
    trunc_hist: Counter = field(default_factory=Counter)
    warnings: int = 0
    errors: list[str] = field(default_factory=list)
    crashes: list[Crash] = field(default_factory=list)
    completion_markers: list[str] = field(default_factory=list)
    loose_stderr: int = 0
    outcome: str = ""
    startup_overhead_s: float | None = None


# --- Parsing ---------------------------------------------------------------------

def parse_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    with path.open("r", errors="replace") as f:
        for line in f:
            if not line.startswith('{"log"'):
                continue
            try:
                d = json.loads(line)
                entries.append(Entry(
                    ts=datetime.fromisoformat(d["time"].replace("Z", "+00:00")),
                    stream=d.get("stream", ""),
                    msg=d["log"].rstrip("\n"),
                ))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                # Skip malformed lines rather than aborting the whole analysis
                continue
    return entries


def classify_exception(final_line: str) -> str:
    for name, pat in EXCEPTION_PATTERNS:
        if re.search(pat, final_line):
            return name
    # Unknown signatures stay visible instead of being silently dropped
    return final_line[:120] if final_line else "unknown (empty traceback)"


def build_runs(entries: list[Entry]) -> tuple[list[Run], int, list[Crash]]:
    """Segment entries into runs and assemble tracebacks statefully."""
    runs: list[Run] = []
    orphan_crashes: list[Crash] = []
    run: Run | None = None
    banner_count = 0
    in_tb = False
    tb_lines: list[str] = []
    tb_ts: datetime | None = None

    def close_tb():
        nonlocal in_tb, tb_lines, tb_ts
        if not in_tb:
            return
        final = next((l.strip() for l in reversed(tb_lines) if l.strip()), "")
        crash = Crash(
            ts=tb_ts, signature=classify_exception(final),
            lines=list(tb_lines),
            episodes_before=len(run.episodes) if run else 0,
        )
        # A traceback before the first run marker (rotated log) is an orphan
        (run.crashes if run else orphan_crashes).append(crash)
        in_tb = False
        tb_lines = []

    for e in entries:
        if e.stream == "stdout" and BANNER_MARKER in e.msg:
            banner_count += 1

        # Traceback state machine (tracebacks live entirely on stderr)
        if e.stream == "stderr" and not in_tb and e.msg.startswith(TRACEBACK_START):
            in_tb, tb_ts, tb_lines = True, e.ts, [e.msg]
            continue
        if in_tb:
            if e.stream == "stderr" and not STRUCTURED_RE.match(e.msg):
                # Chained "During handling..." tracebacks stay in one event
                tb_lines.append(e.msg)
                continue
            close_tb()  # fall through: this line is normal traffic

        # stdout outside tracebacks is entrypoint noise; ignore
        if e.stream != "stderr":
            continue

        # Run boundary (single marker, avoids double counting)
        if RUN_START_MARKER in e.msg:
            m = RUN_CONFIG_RE.search(e.msg)
            run = Run(idx=len(runs), start_ts=e.ts, config=m.group(1) if m else "")
            runs.append(run)
            continue

        # Head of a rotated log before the first run marker
        if run is None:
            run = Run(idx=0, start_ts=e.ts, partial=True)
            runs.append(run)

        structured = STRUCTURED_RE.match(e.msg)
        if not structured:
            run.loose_stderr += 1
            continue

        run.end_ts = e.ts
        level = structured.group(1)

        if level == "ERROR":
            run.errors.append(f"{e.ts.isoformat()} {e.msg.strip()}")
        elif level == "WARNING":
            run.warnings += 1
            if (tm := TRUNC_RE.search(e.msg)):
                run.trunc_hist[int(tm.group(2))] += 1
        elif level == "INFO":
            if (em := EPISODE_RE.search(e.msg)):
                run.episodes.append((int(em.group(1)), float(em.group(3)),
                                     float(em.group(9)), e.ts))
                run.planned_episodes = run.planned_episodes or int(em.group(2))
            elif (sm := TRAINING_START_RE.search(e.msg)):
                run.planned_episodes = int(sm.group(1))
                run.training_start_ts = e.ts
            elif any(h in e.msg.lower() for h in COMPLETION_HINTS):
                run.completion_markers.append(e.msg.strip())

    close_tb()
    return runs, banner_count, orphan_crashes


def finalize(runs: list[Run]) -> None:
    """Classify crash fatality and per-run outcome. Fatal crashes take
    precedence over completion: finishing the episode plan and then dying
    is a crash, not a success."""
    for i, run in enumerate(runs):
        for c in run.crashes:
            # Fatal iff no structured stderr line follows it within this run
            c.fatal = run.end_ts is None or c.ts > run.end_ts

        max_ep = max((n for n, *_ in run.episodes), default=0)
        fatal = [c for c in run.crashes if c.fatal]
        finished_plan = bool(run.planned_episodes and max_ep >= run.planned_episodes)

        if fatal:
            if finished_plan:
                run.outcome = f"crashed-after-final-episode: {fatal[-1].signature}"
            else:
                phase = "startup" if not run.episodes else "mid-training"
                run.outcome = f"crashed-{phase}: {fatal[-1].signature}"
        elif run.completion_markers or finished_plan:
            run.outcome = "completed"
        elif i == len(runs) - 1:
            run.outcome = "open (last run in capture)"
        else:
            run.outcome = "silent-death (ended with no traceback)"
        if run.partial:
            run.outcome = "[partial] " + run.outcome

        if run.training_start_ts and run.start_ts:
            run.startup_overhead_s = (run.training_start_ts - run.start_ts).total_seconds()


# --- Reporting -------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def report(runs: list[Run], banner_count: int, total_entries: int,
           orphan_crashes: list[Crash], show_examples: int):
    span_s = (runs[-1].end_ts or runs[-1].start_ts) - runs[0].start_ts
    total_eps = sum(len(r.episodes) for r in runs)
    total_warnings = sum(r.warnings for r in runs)
    all_crashes = [c for r in runs for c in r.crashes]
    fatal = [c for c in all_crashes if c.fatal]
    survived = [c for c in all_crashes if not c.fatal]
    silent = [r for r in runs if "silent-death" in r.outcome]
    clean = sum(1 for r in runs if r.outcome == "completed")

    print("\n=== Overview ===")
    print(f"Entries parsed:      {total_entries:,}")
    print(f"Wall-clock span:     {fmt_duration(span_s.total_seconds())} "
          f"({runs[0].start_ts:%Y-%m-%d %H:%M} -> {(runs[-1].end_ts or runs[-1].start_ts):%Y-%m-%d %H:%M})")
    print(f"Container runs:      {len(runs)} (CUDA banners on stdout: {banner_count})")
    print(f"Clean exits:         {clean}")
    print(f"Episodes executed:   {total_eps:,}")
    print(f"Warnings:            {total_warnings:,}")
    print(f"Tracebacks:          {len(all_crashes)} total | {len(fatal)} fatal | "
          f"{len(survived)} survived | {len(orphan_crashes)} orphaned by rotation")

    print("\n=== Run outcomes ===")
    for outcome, count in Counter(r.outcome for r in runs).most_common():
        print(f"  {count:4d}  {outcome}")

    if fatal:
        print("\n=== Fatal crash signatures ===")
        sig_counts = Counter(c.signature for c in fatal)
        startup = sum(1 for c in fatal if c.episodes_before == 0)
        print(f"  ({startup} occurred before episode 1 completed; "
              f"{len(fatal) - startup} later)")
        for sig, count in sig_counts.most_common():
            at_eps = [c.episodes_before for c in fatal if c.signature == sig]
            print(f"  {count:4d}  {sig}")
            print(f"        episodes completed before crash: "
                  f"min={min(at_eps)} median={statistics.median(at_eps):.0f} max={max(at_eps)}")

    if survived:
        print("\n=== Survived tracebacks (run continued) ===")
        for sig, count in Counter(c.signature for c in survived).most_common():
            print(f"  {count:4d}  {sig}")

    if silent:
        print("\n=== Silent deaths (no traceback; consistent with SIGKILL/OOM) ===")
        for r in silent:
            print(f"  run {r.idx:3d} | ended {r.end_ts} | "
                  f"episodes done: {len(r.episodes)}")

    all_errors = [e for r in runs for e in r.errors]
    if all_errors:
        print(f"\n=== ERROR lines ({len(all_errors)}) ===")
        for e in all_errors:
            print(f"  {e}")

    eps_counts = [len(r.episodes) for r in runs]
    overheads = [r.startup_overhead_s for r in runs if r.startup_overhead_s is not None]
    gaps = [(runs[i + 1].start_ts - (runs[i].end_ts or runs[i].start_ts)).total_seconds()
            for i in range(len(runs) - 1)]
    ep_deltas = [(b[3] - a[3]).total_seconds()
                 for r in runs for a, b in zip(r.episodes, r.episodes[1:])]

    print("\n=== Per-run statistics ===")
    print(f"  Episodes per run:      min={min(eps_counts)} median={statistics.median(eps_counts):.0f} "
          f"mean={statistics.mean(eps_counts):.1f} max={max(eps_counts)}")
    if overheads:
        print(f"  Startup overhead:      mean={statistics.mean(overheads):.1f}s "
              f"(config load -> training start)")
    if gaps:
        print(f"  Restart gap:           median={statistics.median(gaps):.1f}s "
              f"max={max(gaps):.1f}s (container death -> next config load)")
    if ep_deltas:
        print(f"  Episode duration:      median={statistics.median(ep_deltas):.2f}s "
              f"max={max(ep_deltas):.1f}s (large max = stall/hang)")

    print("\n=== Time accounting ===")
    # Runs that died before "Starting training" still burned config+trace load time
    crashed_in_startup = []
    for r in runs:
        if r.startup_overhead_s is None:
            end = r.end_ts or (r.crashes[-1].ts if r.crashes else r.start_ts)
            crashed_in_startup.append(max(0.0, (end - r.start_ts).total_seconds()))
    wasted_restart_startup = sum(overheads[1:]) if len(overheads) > 1 else 0.0
    redo_eps = sum(len(r.episodes) for r in runs
                   if r.outcome.startswith(("crashed", "silent", "[partial]")))
    redo_time = redo_eps * statistics.median(ep_deltas) if ep_deltas else 0.0
    print(f"  Startup re-runs after first: {fmt_duration(wasted_restart_startup)} "
          f"({len(overheads) - 1 if overheads else 0} restarts)")
    print(f"  Died during startup:         {fmt_duration(sum(crashed_in_startup))} "
          f"across {len(crashed_in_startup)} runs")
    print(f"  Episodes in failed runs:     {redo_eps:,} "
          f"(~{fmt_duration(redo_time)} of training, discarded on restart)")

    trunc = Counter()
    for r in runs:
        trunc.update(r.trunc_hist)
    if trunc:
        top = ", ".join(f"{k}/50: {v}" for k, v in trunc.most_common(5))
        print(f"\n=== Truncation histogram (cloudlets finished, by frequency) ===\n  {top}")

    if show_examples and fatal:
        print("\n=== Example fatal tracebacks (one per top signature) ===")
        seen = set()
        for sig, _ in Counter(c.signature for c in fatal).most_common(show_examples):
            crash = next(c for c in fatal if c.signature == sig and sig not in seen)
            seen.add(sig)
            print(f"\n--- {sig} (crashed after {crash.episodes_before} episodes) ---")
            for line in crash.lines[:30]:
                print(f"  {line}")
            if len(crash.lines) > 30:
                print(f"  ... ({len(crash.lines) - 30} more lines)")


def write_csv(runs: list[Run], path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "start", "end", "duration_s", "config", "planned_episodes",
                    "episodes_done", "last_episode", "last_reward", "last_epsilon",
                    "warnings", "error_lines", "tracebacks", "fatal_tracebacks",
                    "startup_overhead_s", "outcome"])
        for r in runs:
            last = r.episodes[-1] if r.episodes else None
            w.writerow([
                r.idx, r.start_ts.isoformat(),
                r.end_ts.isoformat() if r.end_ts else "",
                round(((r.end_ts or r.start_ts) - r.start_ts).total_seconds(), 1),
                r.config, r.planned_episodes or "", len(r.episodes),
                last[0] if last else "",
                f"{last[1]:.4f}" if last else "",
                f"{last[2]:.4f}" if last else "",
                r.warnings, len(r.errors), len(r.crashes),
                sum(1 for c in r.crashes if c.fatal),
                round(r.startup_overhead_s, 1) if r.startup_overhead_s is not None else "",
                r.outcome,
            ])
    print(f"\nPer-run report written to {path}")


def plot(runs: list[Run]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not installed. Skipping plots.")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    idx = [r.idx for r in runs]
    eps = [len(r.episodes) for r in runs]
    colors = ["green" if r.outcome == "completed" else
              "orange" if "silent-death" in r.outcome else
              "gray" if r.outcome.startswith("open") else "red" for r in runs]
    ax1.bar(idx, eps, color=colors)
    ax1.set_ylabel("Episodes completed")
    ax1.set_title("Episodes per run (green=completed, red=crash, orange=silent death)")
    ax2.plot([r.start_ts for r in runs], eps, marker="o", markersize=3)
    ax2.set_ylabel("Episodes completed")
    ax2.set_title("Run timeline")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Failure analysis for python-agent Docker logs.")
    ap.add_argument("log_file", type=Path)
    ap.add_argument("--csv", type=Path, help="Write per-run report to CSV")
    ap.add_argument("--plot", action="store_true", help="Plot episodes per run (needs matplotlib)")
    ap.add_argument("--examples", type=int, default=3,
                    help="Print one example traceback per top N fatal signatures (0=off)")
    args = ap.parse_args()

    entries = parse_entries(args.log_file)
    if not entries:
        print(f"No parseable entries in {args.log_file}")
        return
    runs, banners, orphan_crashes = build_runs(entries)
    finalize(runs)
    report(runs, banners, len(entries), orphan_crashes, args.examples)
    if args.csv:
        write_csv(runs, args.csv)
    if args.plot:
        plot(runs)


if __name__ == "__main__":
    main()