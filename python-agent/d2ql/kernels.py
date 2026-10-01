# ruff: noqa: N803, N806  -- Triton constexpr/UPPER kernel args are idiomatic.
"""
Custom low-bit matmul kernels (B1 exploration).

The reference "default" kernel is ``torch._int_mm`` (see ``d2ql.precision``),
but it requires ``M > 16`` and ``K``, ``N`` multiples of 8, so a single-sample
RL decision gets zero-padded to a 17-row batch — wasteful at batch=1.

This module provides a hand-written **Triton** int8 matmul kernel that accepts
any ``M`` (no batch padding), pads only ``K``/``N`` inside the kernel blocks, and
dequantizes with a per-output-channel scale. We iterate up to a handful of
tuning configurations and compare each against the ``int_mm`` baseline.

Kernel contract
---------------
``triton_int8_matmul(a, b, scale) -> y``
    ``a``: int8 [M, K],  ``b``: int8 [N, K],  ``scale``: fp32 [N]
    ``y``: fp32 [M, N] where y[m, n] = (sum_k a[m,k] * b[n,k]) * scale[n]
"""

from __future__ import annotations

import logging

import numpy as _np
import torch
import triton
import triton.language as tl

from d2ql.precision import quantize_int8

logger = logging.getLogger(__name__)


@triton.jit
def _int8_mm_kernel(
    a_ptr, b_ptr, out_ptr, scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_b = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        acc += tl.dot(a, b)

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    y = acc.to(tl.float32) * scale[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Tuning grid explored by `benchmark_custom_kernel` (bounded iterations). Only
# BLOCK_M / BLOCK_N / num_warps are tuned: BLOCK_K is forced to the full K and
# num_stages to 1 (single-K-iteration, see triton_int8_matmul).
TRITON_CONFIGS = [
    dict(BLOCK_M=16, BLOCK_N=32, num_warps=4),
    dict(BLOCK_M=16, BLOCK_N=64, num_warps=4),
    dict(BLOCK_M=32, BLOCK_N=32, num_warps=4),
    dict(BLOCK_M=64, BLOCK_N=64, num_warps=8),
    dict(BLOCK_M=16, BLOCK_N=64, num_warps=2),
]


def triton_int8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    cfg: dict | None = None,
) -> torch.Tensor:
    """y[M, N] = (a @ b^T) * scale[N], all int8 -> fp32 (custom Triton kernel).

    ``BLOCK_K`` is always forced to ``>= K`` so the kernel runs a SINGLE K
    iteration: Triton 3.2's ``tl.dot`` int8 producing ``acc += dot(...)`` across
    multiple K blocks is buggy (verified empirically), while a single big dot is
    exact. ``num_stages`` is set to 1 since there is nothing to pipeline.
    """
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert a.is_contiguous() and b.is_contiguous()
    assert a.is_cuda, "Triton kernel requires CUDA"
    M, K = a.shape
    N = b.shape[0]
    assert b.shape[1] == K

    out = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if cfg is None:
        cfg = dict(BLOCK_M=16, BLOCK_N=32, num_warps=4)
    block_k = max(16, K)  # single K-iteration (correctness, avoids tl.dot K-bug)

    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _int8_mm_kernel[grid](
        a, b, out, scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=block_k,
        GROUP_M=cfg.get("GROUP_M", 1), num_warps=cfg["num_warps"], num_stages=1,
    )
    return out


def lowbit_matmul_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cfg: dict | None = None,
) -> torch.Tensor:
    """Full custom-kernel path: quantize -> Triton int8 matmul -> dequant -> bias."""
    x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
    qx, sx = quantize_int8(x_2d, channel_dim=None)        # [M, in] int8, scalar
    qw, sw = quantize_int8(weight, channel_dim=0)         # [out, in] int8, [out, 1]
    scale = (sx.float() * sw.float()).squeeze(-1).contiguous()   # [out]
    y = triton_int8_matmul(qx, qw, scale, cfg=cfg)
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# CPU 1-bit / ternary bit-packed kernels (B1, deployment target = CPU)
#
# P3 decision: int8 is retired from the CPU sweep and replaced by ternary. An
# int8 x int8 CPU GEMM has no fast path on this host (Zen 3 has no VNNI), so it
# can only be emulated (int32) and is always slower than fp32. Ternary, by
# contrast, has a genuine bit-packed kernel: a ternary weight matrix splits into
# two binary matrices (w_pos, w_neg) and each pass reduces to XOR + popcount on
# the integer ALU (see `ternary_matmul_batched`).
#
# These are the classic Binary/Ternary Neural Network kernels: instead of
# floating-point multiply-accumulate, we pack 32 weight/activation signs into
# one uint32 word and reduce the inner product with XOR + popcount. Pure
# bitwise arithmetic — genuinely faster than fp32 on CPU integer ALUs (and the
# only way low precision can win there, since torch._int_mm is CUDA-only and
# PyTorch's CPU int8 accel is qint8-only). Intended for BATCH inference: per
# sample the packing overhead dominates, so these shine via `samples/sec`.
# ---------------------------------------------------------------------------


def pack_binary_bits(x: "torch.Tensor | _np.ndarray", K_out: "int | None" = None) -> _np.ndarray:
    """Binarize last dim to {+1,-1} and pack 32 signs per uint32 word -> [.., W].

    Bit j within a word holds element (32*w + j), LSB-first. ``K_out`` pads the
    last word to a full 32 (used for reference/training where K may not be a
    multiple of 32); caller is responsible for masking the padding in the dot.
    """
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    arr = _np.asarray(arr)
    K = K_out if K_out is not None else arr.shape[-1]
    W = -(-K // 32)  # ceil(K/32)
    flat = _np.zeros((_np.prod(arr.shape[:-1], dtype=int), K), dtype=arr.dtype)
    flat[:, : arr.shape[-1]] = arr.reshape(-1, arr.shape[-1])
    words = _np.zeros((flat.shape[0], W), dtype=_np.uint32)
    shifts = (1 << _np.arange(32, dtype=_np.uint32))
    for w in range(W):
        lo = w * 32
        hi = min(lo + 32, K)
        chunk = (flat[:, lo:hi] > 0).astype(_np.uint32)  # +1 -> bit 1, -1 -> bit 0
        col = _np.zeros((flat.shape[0], 32), dtype=_np.uint32)
        col[:, : hi - lo] = chunk
        words[:, w] = (col * shifts[None, :]).sum(axis=1, dtype=_np.uint32)
    return words


def word_masks(K: int) -> _np.ndarray:
    """Per-word masks (LSB-first) marking the real (non-padded) bit positions."""
    W = -(-K // 32)
    masks = _np.zeros(W, dtype=_np.uint64)
    for w in range(W):
        real = min(32, K - 32 * w)
        if real > 0:
            masks[w] = _np.uint64((1 << real) - 1)
    return masks


def binary_matmul_batched(x_words, w_words, masks) -> _np.ndarray:
    """y[M,N] = dot of binary (±1) vectors from packed words, bitwise + popcount.

    ``x_words`` [M, W], ``w_words`` [N, W], ``masks`` [W]. Non-padded spurious
    matches are masked out. dot(m,n) = sum_w ( (2*real_matches_w) - real_w ).
    """
    xu = _np.asarray(x_words, dtype=_np.uint64)        # [M, W]
    wu = _np.asarray(w_words, dtype=_np.uint64)       # [N, W]
    masks64 = _np.asarray(masks, dtype=_np.uint64)[None, None, :]  # [1, 1, W]
    xor = (xu[:, None, :] ^ wu[None, :, :]) & masks64  # [M, N, W]
    cnt = _np.bitwise_count(xor)                        # [M, N, W] MISMATCHES-per-word
    real = _np.bitwise_count(masks64)                   # [1, 1, W] real bits per word
    out = _np.sum(real.astype(_np.int64) - 2 * cnt.astype(_np.int64), axis=2)
    return out


def binary_network_forward(x, layer_weights: list) -> _np.ndarray:
    """Full MLP in 1-bit: binarize activations between layers; weights pre-packed.

    ``x``: fp [M, K]; ``layer_weights``: list of (w_bits [N, W], bias [N], masks [W]).
    Returns raw final dot activations [M, A] (pre-argmax). Hard-binarizes the
    input to every linear layer — including the first — as is standard in BNNs.
    """
    act = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    act = _np.asarray(act)
    for i, (wbits, bias, masks) in enumerate(layer_weights):
        xbits = pack_binary_bits(act)                       # activations -> ±1 packed
        out = binary_matmul_batched(xbits, wbits, masks)    # [M, N]
        if bias is not None:
            out = out + bias.detach().cpu().numpy()[None, :]
        if i < len(layer_weights) - 1:
            act = _np.where(out > 0, 1.0, -1.0)             # hard binarize hidden layers
    return out  # final layer returns raw dots (pre-argmax)


# ---------------------------------------------------------------------------
# Ternary bit-packed kernel (P3): weights in {-s, 0, +s} via TWO binary passes
#
# A ternary weight matrix is decomposed as w = s * (w_pos - w_neg), where
# w_pos = 1{w > 0} and w_neg = 1{w < 0} are binary and s is the per-output-
# channel scale. With activations hard-binarized to ±1 (standard for bit-packed
# BNN kernels), the exact identity
#
#     dot(x, w_pos) - dot(x, w_neg) = 2 * dot(x, w)
#
# (because each ±1 encoding double-counts) means the ternary matmul costs only
# two XOR+popcount passes plus a shift. No float or integer multiply is needed,
# which is exactly the regime where a CPU without VNNI can beat fp32.
# ---------------------------------------------------------------------------


def ternary_matmul_batched(
    x: "torch.Tensor | _np.ndarray",
    weight: "torch.Tensor | _np.ndarray",
    bias: "torch.Tensor | _np.ndarray | None" = None,
) -> _np.ndarray:
    """y[M, N] = (x_binarized @ w^T) * channel_scale + bias, via 2 binary passes.

    ``x``: fp [M, K] (binarized internally by sign), ``weight``: fp [N, K] holding
    ternary values ``{-s, 0, +s}`` per output channel, ``bias``: [N] or None.
    Returns fp64 [M, N] — the caller casts to fp32.

    Activations are hard-binarized to ±1 (sign), matching the classic
    BNN/ternary bit-packed formulation, so this is a *deployment* approximation
    of the 8-bit-activation STE training path — not a bit-exact equivalent.
    """
    act = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    w_np = weight.detach().cpu().numpy() if isinstance(weight, torch.Tensor) else weight
    act = _np.asarray(act, dtype=_np.float64)
    w_np = _np.asarray(w_np, dtype=_np.float64)

    if act.ndim == 1:
        act = act[None, :]
    if act.ndim > 2:
        act = act.reshape(-1, act.shape[-1])

    K = w_np.shape[1]
    masks = word_masks(K)

    # Per-output-channel scale, taken from the largest-magnitude ternary value.
    scale = _np.abs(w_np).max(axis=1)
    scale = _np.where(scale > 0.0, scale, 1.0)

    # Binary planes of the ternary weights: w = s * (w_pos - w_neg).
    w_pos_words = pack_binary_bits((w_np > 0.0).astype(_np.float64))
    w_neg_words = pack_binary_bits((w_np < 0.0).astype(_np.float64))

    # Chunk over rows: binary_matmul_batched materializes an [M, N, W] temporary,
    # so batch x width x words can reach gigabytes for large hidden sizes. Cap
    # the intermediate to ~2e7 elements (~160 MB) regardless of the batch.
    n_out = w_np.shape[0]
    words = w_pos_words.shape[1]
    max_elems = 2.0e7
    chunk = max(1, int(max_elems / max(n_out * words, 1)))

    y = _np.empty((act.shape[0], n_out), dtype=_np.float64)
    for start in range(0, act.shape[0], chunk):
        block = _np.where(act[start:start + chunk] > 0.0, 1.0, -1.0)
        x_words = pack_binary_bits(block)
        pos = binary_matmul_batched(x_words, w_pos_words, masks).astype(_np.float64)
        neg = binary_matmul_batched(x_words, w_neg_words, masks).astype(_np.float64)
        # The ±1 encodings double-count, hence the 0.5 factor (see identity above).
        y[start:start + chunk] = 0.5 * (pos - neg) * scale[None, :]

    if bias is not None:
        b = bias.detach().cpu().numpy() if isinstance(bias, torch.Tensor) else bias
        if b is not None:
            y = y + _np.asarray(b, dtype=_np.float64)[None, :]
    return y

