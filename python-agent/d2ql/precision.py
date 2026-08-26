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
