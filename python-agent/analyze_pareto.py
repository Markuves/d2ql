#!/usr/bin/env python
"""Pareto analysis of the H4 precision x capacity sweep (P1).

Reads the aggregate results CSV produced by ``main.py`` and reports, for a chosen
quality metric:

  * per-cell aggregation over seeds (mean +- std),
  * the non-dominated set on several *normalized* cost axes,
  * the throughput-vs-batch curve per precision (the regime where a low-bit
    kernel could actually win).

Why normalize the cost axis
---------------------------
Raw latency mixes three confounds at once: bit-width, device and capacity. FLOPs
and effective-capacity-bits are bit-width invariant (the dense matmul cost does
not change when you round the weights), so plotting quality against them leaves
any residual difference attributable to precision alone:

    flops                    = 2 * MACs per forward  (compute cost / decision)
    effective_capacity_bits  = n_params * bits       (model footprint in bits)
    packed_size_mb           = n_params * bits / 8   (theoretical artifact size)

Usage
-----
    python analyze_pareto.py --results outputs/results/h4_results.csv
    python analyze_pareto.py --quality norm_reward --out outputs/results/pareto.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COST_AXES = [
    ("flops", "FLOPs/decisión"),
    ("effective_capacity_bits", "capacidad efectiva (params×bits)"),
    ("packed_size_mb", "tamaño empaquetado (MB)"),
    ("latency_mean_ms", "latencia inferencia (ms)"),
]
GROUP_KEYS = ["precision", "device", "hidden_size"]


def _md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = ".4g") -> str:
    """Render a DataFrame as a GitHub markdown table (no `tabulate` dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if isinstance(v, float):
                cells.append("nan" if v != v else format(v, floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Results CSV is empty: {path}")
    # Numeric coercion for the columns we aggregate / plot.
    for col in [
        "flops", "effective_capacity_bits", "packed_size_mb", "latency_mean_ms",
        "latency_p95_ms", "throughput_pps", "params", "bits", "seed",
        "eval_mean_reward", "norm_reward", "eval_makespan", "eval_sla_rate",
        "wall_clock_s",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pareto_front(df: pd.DataFrame, cost: str, quality: str) -> pd.DataFrame:
    """Rows not dominated on (minimize cost, maximize quality)."""
    pts = df[[cost, quality]].to_numpy(dtype=float)
    keep = []
    for i, (c, q) in enumerate(pts):
        dominated = False
        for j, (c2, q2) in enumerate(pts):
            if i == j:
                continue
            if c2 <= c and q2 >= q and (c2 < c or q2 > q):
                dominated = True
                break
        keep.append(not dominated)
    return df[keep]


def aggregate(df: pd.DataFrame, quality: str) -> pd.DataFrame:
    """Per (precision, device, hidden): mean/std over seeds of the key metrics."""
    cols = [quality, "latency_mean_ms", "throughput_pps", "eval_makespan", "wall_clock_s"]
    cols = [c for c in cols if c in df.columns]
    agg = df.groupby(GROUP_KEYS)[cols].agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    n_seeds = df.groupby(GROUP_KEYS)["seed"].nunique().rename("n_seeds")
    for det in ["flops", "effective_capacity_bits", "packed_size_mb", "params", "bits"]:
        if det in df.columns:
            det_mean = df.groupby(GROUP_KEYS)[det].mean()
            agg[det] = det_mean
    return agg.join(n_seeds).reset_index()


def throughput_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Wide table of mean samples/sec per (precision, device) x batch."""
    if "throughput_by_batch" not in df.columns:
        return None
    records = []
    for _, row in df.iterrows():
        raw = row.get("throughput_by_batch")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for batch, vals in parsed.items():
            records.append({
                "precision": row["precision"],
                "device": row["device"],
                "batch": int(batch),
                "samples_per_sec": float(vals.get("samples_per_sec", float("nan"))),
            })
    if not records:
        return None
    tdf = pd.DataFrame(records)
    return (
        tdf.groupby(["precision", "device", "batch"])["samples_per_sec"]
        .mean()
        .unstack("batch")
        .round(0)
    )


def render(df: pd.DataFrame, quality: str) -> str:
    lines: list[str] = []
    out = lines.append
    n_seeds = int(df["seed"].nunique()) if "seed" in df.columns else 1
    out(f"# Pareto H4 — calidad = `{quality}`\n")
    out(f"- filas: {len(df)} | semillas distintas: {n_seeds} "
        f"| celdas (precisión×device×ancho): {df.groupby(GROUP_KEYS).ngroups}\n")

    agg = aggregate(df, quality)
    q_mean = f"{quality}_mean"
    q_std = f"{quality}_std"

    out("## Agregado por celda (media ± std sobre semillas)\n")
    show = GROUP_KEYS + [c for c in [q_mean, q_std, "latency_mean_ms_mean",
                                     "flops", "packed_size_mb", "n_seeds"] if c in agg.columns]
    out(_md_table(agg[show], floatfmt=".4g"))
    out("")

    if "seed" in df.columns and n_seeds < 3:
        out("> Aviso: <3 semillas. La std de arriba es del mismo orden que el "
            "efecto medido; el frente es orientativo, no concluyente.\n")

    # Pareto fronts on normalized cost axes (lower cost, higher quality).
    cell = agg.rename(columns={q_mean: "quality"}).dropna(subset=["quality"])
    for cost, label in COST_AXES:
        # Aggregated metrics carry a "_mean" suffix; deterministic ones do not.
        actual = cost if cost in cell.columns else (
            f"{cost}_mean" if f"{cost}_mean" in cell.columns else None
        )
        if actual is None:
            out(f"## Frente por {label}\n\n_(columna `{cost}` ausente)_\n")
            continue
        sub = cell[[*GROUP_KEYS, actual, "quality"]].dropna()
        front = pareto_front(sub, actual, "quality").sort_values(actual)
        out(f"## Frente no dominado por {label} (`{actual}`)\n")
        if front.empty:
            out("_(sin puntos con datos suficientes)_\n")
        else:
            out(_md_table(front, floatfmt=".4g"))
            out("")

    tt = throughput_table(df)
    out("## Throughput medio (samples/s) vs batch\n")
    if tt is None:
        out("_(columna `throughput_by_batch` ausente: resultados previos a P1)_\n")
    else:
        out(_md_table(tt, index=True, floatfmt=".0f"))
        out("")
        out("Un cruce entre filas a batch creciente es lo único que demuestra una "
            "ventaja real de kernel; si ninguna curva cruza a fp32, la ventaja de "
            "la baja precisión es de tamaño, no de velocidad.\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto analysis for the H4 sweep")
    parser.add_argument(
        "--results", default="outputs/results/h4_results.csv",
        help="Path to the aggregate results CSV",
    )
    parser.add_argument(
        "--quality", default="eval_mean_reward",
        choices=["eval_mean_reward", "norm_reward"],
        help="Quality metric for the front",
    )
    parser.add_argument("--out", default="", help="Optional path to also write the markdown")
    args = parser.parse_args()

    df = load_results(Path(args.results))
    report = render(df, args.quality)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
