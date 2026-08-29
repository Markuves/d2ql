"""
Compare custom low-bit kernels against the default ``torch._int_mm`` baseline
over the real 3-layer Q-network, measuring batched throughput (samples/sec).

This is the bounded "N iterations" attempt at writing a faster kernel: it sweeps
a handful of Triton tile/warp configs (TRITON_CONFIGS in d2ql/kernels.py) and
compares each against:

  * fp32        : plain nn.Linear network (the accuracy baseline).
  * int_mm      : torch._int_mm deploy path (the "default" low-bit kernel).
  * triton[c]   : our custom Triton kernel, one pass per config.

Usage:
    uv run python scripts/benchmark_kernels.py [--batch 1,32,128,512] [--iter N]

Result: a table of samples/sec and speedup-vs-fp32 per backend per batch,
plus the best Triton config. Nothing here touches training.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from d2ql.kernels import TRITON_CONFIGS, lowbit_matmul_triton
from d2ql.precision import lowbit_real_matmul


def _relu(x):
    return torch.relu(x)


def make_weights(device, state_dim=9, hidden=256, action=4, seed=0):
    torch.manual_seed(seed)
    w1 = (torch.randn(hidden, state_dim, device=device) - 0.5) * 2
    b1 = torch.randn(hidden, device=device) * 0.1
    w2 = (torch.randn(hidden, hidden, device=device) - 0.5) * 2
    b2 = torch.randn(hidden, device=device) * 0.1
    w3 = (torch.randn(action, hidden, device=device) - 0.5) * 2
    b3 = torch.randn(action, device=device) * 0.1
    return [(w1, b1), (w2, b2), (w3, b3)]


def layers_fp32(device, state_dim=9, hidden=256, action=4, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, action),
    ).to(device).eval()
    return net


def forward_int8(layers, x, backend_fn):
    for (w, b) in layers:
        x = backend_fn(x, w, b)
        x = _relu(x)
    return x


def timeit(fn, warmup=5, runs=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def benchmark(device, batches, iters, state_dim=9, hidden=256, action=4):
    layers = make_weights(device, state_dim, hidden, action)
    net = layers_fp32(device, state_dim, hidden, action)

    rows = []
    for bsz in batches:
        x = torch.zeros(bsz, state_dim, device=device, dtype=torch.float32)

        # fp32 baseline
        ms_fp32 = timeit(lambda: net(x), runs=iters)

        # int_mm default kernel (custom int8 kernels do not use torch._int_mm here)
        ms_intmm = timeit(
            lambda: forward_int8(layers, x, lowbit_real_matmul), runs=iters
        )

        best = None
        for c in TRITON_CONFIGS:
            ms = timeit(
                lambda c=c: forward_int8(layers, x, lambda a, w, b: lowbit_matmul_triton(a, w, b, c)),
                runs=iters,
            )
            row = {
                "batch": bsz,
                "ms": ms,
                "pps": bsz * 1000.0 / ms,
                "vs_fp32": ms_fp32 / ms,
                "cfg": dict(c),
            }
            rows.append(("triton", row))
            if best is None or ms < best[1]["ms"]:
                best = c, row

        rows.append(("fp32", {"batch": bsz, "ms": ms_fp32, "pps": bsz * 1000.0 / ms_fp32, "vs_fp32": 1.0, "cfg": None}))
        rows.append(("int_mm", {"batch": bsz, "ms": ms_intmm, "pps": bsz * 1000.0 / ms_intmm, "vs_fp32": ms_fp32 / ms_intmm, "cfg": "torch._int_mm"}))
        rows.append(("TRITON_BEST", {"batch": bsz, **{k: v for k, v in best[1].items() if k not in ("batch",)}}))
        torch.cuda.empty_cache()

    return rows


def main():
    parser = argparse.ArgumentParser(description="Kernel benchmark: custom Triton vs int_mm vs fp32")
    parser.add_argument("--batch", type=str, default="1,32,128,512")
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()
    batches = [int(x) for x in args.batch.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Triton kernel requires CUDA; aborting.")
        return

    print(f"device={device}  batches={batches}  iters={args.iter}")
    print(f"{'batch':>6} | {'backend':<12} | {'ms/call':>9} | {'samples/s':>12} | {'vs fp32':>8} | cfg")
    print("-" * 90)
    for kind, row in benchmark(device, batches, args.iter):
        cfg = row.get("cfg")
        cfgstr = (str(cfg) if cfg is not None else "") if not isinstance(cfg, dict) else f"M{cfg['BLOCK_M']}/N{cfg['BLOCK_N']}/w{cfg['num_warps']}"
        print(f"{row['batch']:>6} | {kind:<12} | {row['ms']:9.3f} | {row['pps']:12.0f} | {row['vs_fp32']:8.2f}x | {cfgstr}")


if __name__ == "__main__":
    main()
