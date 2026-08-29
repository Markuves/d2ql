"""
Native bit-width training (H4).

The Q-network is instantiated and updated at a chosen precision.

  32: torch.float32 parameters and activations.
  16: torch.float16 parameters and activations (no AMP / GradScaler).
   8 / 4: signed integer grid (symmetric, per-channel weights).
  ternary: 1.58-bit BitNet-style weights in {-1, 0, +1} (log2(3) ≈ 1.58).
    Activations stay 8-bit when quantize_activations is true.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PRECISION_BITS = {
    "ternary": 1.58,
    "4": 4,
    "8": 8,
    "16": 16,
    "32": 32,
}

GRID_PRECISIONS = frozenset({"ternary", "4", "8"})


def parse_precision(value: object) -> str:
    """Resolve agent.precision to one of: ternary, 4, 8, 16, 32."""
    if value in (1.58, "1.58"):
        return "ternary"
    if isinstance(value, float) and abs(value - 1.58) < 1e-6:
        return "ternary"
    if isinstance(value, int):
        key = str(value)
    else:
        key = str(value).strip().lower()
    if key not in PRECISION_BITS:
        raise ValueError(
            f"Unknown precision '{value}'. Use one of: ternary, 4, 8, 16, 32."
        )
    return key


def precision_bits(name: str) -> float:
    return PRECISION_BITS[name]


def compute_dtype(name: str) -> torch.dtype:
    if name == "16":
        return torch.float16
    return torch.float32


def _channel_scale(x: torch.Tensor, channel_dim: Optional[int], reduce: str) -> torch.Tensor:
    if channel_dim is None:
        mag = x.detach().abs()
        return (mag.mean() if reduce == "mean" else mag.amax()).clamp(min=1e-8)
    reduce_dims = tuple(i for i in range(x.ndim) if i != channel_dim)
    mag = x.detach().abs()
    if reduce == "mean":
        return mag.mean(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    return mag.amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)


def quantize_ternary(x: torch.Tensor, channel_dim: Optional[int] = None) -> torch.Tensor:
    """Abs-mean ternary quant: weights snap to {-scale, 0, +scale}."""
    scale = _channel_scale(x, channel_dim, reduce="mean")
    return torch.round(x / scale).clamp(-1, 1) * scale


# ---------------------------------------------------------------------------
# Real low-bit kernel (B1 fix, option 1)
#
# NativeBitLinear.B1: on CUDA the grid precisions (ternary / 4 / 8) run a
# GENUINE int8 matmul kernel via torch._int_mm (int8 x int8 -> int32), which
# is actually faster than float32 on tensor-core GPUs. Weights are already
# projected onto their grid (so a ternary weight still only has 3 distinct
# values); they are only *represented* as int8 for the fast kernel. This makes
# the measured inference latency reflect real low-bit speedup instead of the
# fake float32-rounded path used during training (which needs gradients).
# ---------------------------------------------------------------------------

def _int8_binary_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [M,K] int8, b: [K,N] int8 -> [M,N] int32 (real int8 tensor-core kernel).

    torch._int_mm requires M > 16 and K, N as multiples of 8 (built for
    LLM-style batching). RL acts on single samples with arbitrary widths, so we
    zero-pad M / K / N to the kernel constraints and slice the result back —
    the matmul is still a genuine int8 tensor-core operation.
    """
    if a.is_cuda:
        m, k = a.shape
        n = b.shape[1]
        pm = max(m, 17)
        pk = ((k + 7) // 8) * 8
        pn = ((n + 7) // 8) * 8
        if pm != m or pk != k:
            a = F.pad(a, (0, pk - k, 0, pm - m))
        if pk != k or pn != n:
            b = F.pad(b, (0, pn - n, 0, pk - k))
        y = torch._int_mm(a.contiguous(), b.contiguous())
        return y[:m, :n]
    # CPU fallback (correct, but no tensor-core speedup): emulate int arithmetic.
    return (a.to(torch.int32) @ b.to(torch.int32))


def quantize_int8(x: torch.Tensor, channel_dim: Optional[int], reduce: str = "amax"):
    """Symmetric signed int8 quant. Returns (q: int8, scale: fp32 scalar or [.,1])."""
    qmax = 127
    scale = _channel_scale(x, channel_dim, reduce) / qmax
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def lowbit_real_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Genuine int8 x int8 matmul with per-output-channel weight scale.

    x: [..., in] fp32, weight: [out, in] fp32 (grid-projected), bias: [out].
    Returns y: [..., out] fp32.
    """
    x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
    qx, sx = quantize_int8(x_2d, channel_dim=None)           # [batch, in], scalar
    qw, sw = quantize_int8(weight, channel_dim=0)            # [out, in], [out, 1]
    y_int = _int8_binary_matmul(qx, qw.t())                  # [batch, out] int32
    scale_out = (sx.float() * sw.float()).t()                # [1, out]
    y = (y_int.float() * scale_out)                          # [batch, out] fp32
    if bias is not None:
        y = y + bias
    return y


def model_macs(
    state_dim: int,
    hidden_size: int,
    n_hidden_layers: int,
    action_dim: int,
) -> int:
    """Multiply-accumulate operations for a single forward pass (batch=1, B2).

    Each MLP layer (in -> out) costs `in*out` MACs; feeds the FLOPs metric that
    normalizes capacity across precision x hidden-size combinations so the
    Pareto comparison compares real compute cost, not raw neuron count.
    """
    macs = state_dim * hidden_size
    for _ in range(max(n_hidden_layers - 1, 0)):
        macs += hidden_size * hidden_size
    macs += hidden_size * action_dim
    return macs


def model_flops(state_dim: int, hidden_size: int, n_hidden_layers: int, action_dim: int) -> float:
    """FLOPs per forward pass = 2 * MACs (multiply + add)."""
    return 2.0 * model_macs(state_dim, hidden_size, n_hidden_layers, action_dim)


def effective_capacity_bits(n_params: int, bits: float) -> float:
    """B2: parameter count weighted by stored bit-width (n_params * bits)."""
    return n_params * bits


def quantize_symmetric(
    x: torch.Tensor,
    bits: int,
    channel_dim: Optional[int] = None,
) -> torch.Tensor:
    """Snap tensor onto a signed symmetric integer grid of `bits` bits."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = _channel_scale(x, channel_dim, reduce="amax") / qmax
    return torch.round(x / scale).clamp(qmin, qmax) * scale


def quantize_to_grid(
    x: torch.Tensor,
    precision: str,
    channel_dim: Optional[int] = None,
) -> torch.Tensor:
    if precision == "ternary":
        return quantize_ternary(x, channel_dim)
    return quantize_symmetric(x, int(precision), channel_dim)


class QuantizeSTE(torch.autograd.Function):
    """Straight-through estimator: discrete forward, identity backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, precision: str, channel_dim: Optional[int]):
        return quantize_to_grid(x, precision, channel_dim)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class NativeBitLinear(nn.Module):
    """Linear layer whose weights live on a 4-bit, 8-bit, or ternary grid."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        precision: str,
        bias: bool = True,
        quantize_activations: bool = True,
    ):
        super().__init__()
        if precision not in GRID_PRECISIONS:
            raise ValueError("NativeBitLinear is for ternary, 4, and 8 only.")
        self.precision = precision
        self.quantize_activations = quantize_activations
        # BitNet-style: ternary weights, 8-bit activations.
        self.activation_precision = "8" if precision == "ternary" else precision
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        self.project()
        # When True, forward() uses the real int8 matmul kernel (B1) instead of
        # the fake-quant STE path. Enabled for inference / latency benchmarks.
        self.deploy = False

    def project(self) -> None:
        """Force stored parameters onto the target grid."""
        with torch.no_grad():
            self.weight.data.copy_(
                quantize_to_grid(self.weight.data, self.precision, channel_dim=0)
            )
            if self.bias is not None:
                self.bias.data.copy_(
                    quantize_to_grid(self.bias.data, self.precision, channel_dim=None)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            # B1: real int8 tensor-core kernel for inference / latency benchmark.
            with torch.no_grad():
                return lowbit_real_matmul(x, self.weight, self.bias)
        weight = QuantizeSTE.apply(self.weight, self.precision, 0)
        bias = (
            QuantizeSTE.apply(self.bias, self.precision, None)
            if self.bias is not None
            else None
        )
        if self.quantize_activations:
            x = QuantizeSTE.apply(x, self.activation_precision, None)
        return F.linear(x, weight, bias)


def project_native_parameters(module: nn.Module) -> None:
    """Re-snap every NativeBitLinear after an optimizer step."""
    for child in module.modules():
        if isinstance(child, NativeBitLinear):
            child.project()


def packed_size_mb(n_params: int, name: str) -> float:
    return (n_params * precision_bits(name)) / 8.0 / 1e6


def lookup_max_hidden(mapping: dict, name: str):
    """Resolve max_hidden_size for a precision; YAML keys may be int or str."""
    if not mapping:
        return None
    bits = precision_bits(name)
    candidates = [name, bits]
    if name.isdigit():
        candidates.append(int(name))
    candidates.append(str(bits))
    for key in candidates:
        if key in mapping:
            return int(mapping[key])
    return None


def h4_capacity_plan(bits_list, hidden_sizes, max_hidden_size: dict) -> list[tuple[str, int]]:
    """
    Capacity-major schedule: every precision starts at the same hidden width,
    then larger widths drop higher-precision models.
    """
    plan: list[tuple[str, int]] = []
    for hidden in hidden_sizes:
        hidden = int(hidden)
        for raw in bits_list:
            name = parse_precision(raw)
            cap = lookup_max_hidden(max_hidden_size, name)
            if cap is not None and hidden > cap:
                continue
            plan.append((name, hidden))
    return plan
