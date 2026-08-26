"""
H2: Post-Training Quantization

Applies dynamic quantization (INT8) and half-precision (FP16) to a trained
DDQN checkpoint and benchmarks inference latency and model size reduction
against the FP32 baseline. No retraining is required.

Usage (standalone):
    uv run python -m d2ql.quantization \
        --checkpoint outputs/checkpoints/checkpoint_final.pt \
        --output-dir outputs/checkpoints/quantized \
        --state-dim 9 \
        --action-dim 4
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Mirror of the QNetwork in agent.py — must stay in sync."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_online_net(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> QNetwork:
    """Load only the online network weights from a DDQNAgent checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net = QNetwork(state_dim, action_dim).to(device)
    net.load_state_dict(checkpoint["online_net"])
    net.eval()
    logger.info("Loaded online network from %s.", checkpoint_path)
    return net


def model_size_mb(model: nn.Module) -> float:
    """Return the in-memory size of model parameters in MB."""
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    total_bytes += sum(
        b.nelement() * b.element_size() for b in model.buffers()
    )
    return total_bytes / (1024 ** 2)


def benchmark_latency(
    model: nn.Module,
    state_dim: int,
    device: torch.device,
    n_warmup: int = 50,
    n_runs: int = 500,
) -> float:
    """
    Measure average single-sample inference latency in milliseconds.
    Warmup runs are discarded to avoid JIT and cache cold-start effects.
    """
    dummy = torch.rand(1, state_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0  # ms per inference


def quantize_dynamic_int8(model: QNetwork) -> nn.Module:
    """
    Apply PyTorch dynamic INT8 quantization to all Linear layers.
    Dynamic quantization computes scale factors at runtime from actual
    activations rather than requiring a calibration dataset. This makes
    it well suited to the DDQN online network where activation distributions
    shift during deployment.
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    logger.info("Applied dynamic INT8 quantization.")
    return quantized


def quantize_fp16(model: QNetwork, device: torch.device) -> nn.Module:
    """
    Cast model weights to FP16 (half precision).
    FP16 is supported natively on CUDA devices with Tensor Cores and
    provides approximately 2x memory reduction with minimal accuracy loss
    for inference. Falls back gracefully to CPU if no GPU is available.
    """
    if device.type != "cuda":
        logger.warning(
            "FP16 quantization is most effective on CUDA devices. "
            "Running on CPU will not benefit from Tensor Core acceleration."
        )
    fp16_model = model.half().to(device)
    logger.info("Cast model to FP16.")
    return fp16_model


def evaluate_accuracy_delta(
    fp32_net: QNetwork,
    quantized_net: nn.Module,
    state_dim: int,
    device: torch.device,
    n_samples: int = 1000,
    fp16: bool = False,
) -> float:
    """
    Estimate accuracy delta between FP32 and quantized network by comparing
    greedy action selections on random inputs. Returns the fraction of samples
    where both networks agree on the argmax action.
    """
    fp32_net.eval()
    quantized_net.eval()

    agree = 0
    with torch.no_grad():
        for _ in range(n_samples):
            state = torch.rand(1, state_dim, dtype=torch.float32, device=device)

            q_fp32 = fp32_net(state)
            action_fp32 = int(q_fp32.argmax(dim=1).item())

            if fp16:
                q_quant = quantized_net(state.half())
            else:
                q_quant = quantized_net(state.cpu())

            action_quant = int(q_quant.argmax(dim=1).item())

            if action_fp32 == action_quant:
                agree += 1

    agreement_rate = agree / n_samples
    logger.info("Action agreement rate FP32 vs quantized: %.4f", agreement_rate)
    return agreement_rate


def run_quantization(
    checkpoint_path: str,
    output_dir: str,
    state_dim: int,
    action_dim: int,
) -> dict:
    """
    Full H2 quantization pipeline. Returns a results dict with size and
    latency benchmarks for all three precision levels.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running quantization on device: %s", device)

    # Load FP32 baseline
    fp32_net = load_online_net(checkpoint_path, state_dim, action_dim, device)

    results: dict = {}

    # ------------------------------------------------------------------
    # FP32 baseline measurements
    # ------------------------------------------------------------------
    fp32_size = model_size_mb(fp32_net)
    fp32_latency = benchmark_latency(fp32_net, state_dim, device)
    results["fp32"] = {
        "size_mb": fp32_size,
        "latency_ms": fp32_latency,
        "agreement": 1.0,
    }
    logger.info("FP32 | Size: %.3f MB | Latency: %.4f ms", fp32_size, fp32_latency)

    # ------------------------------------------------------------------
    # INT8 dynamic quantization
    # ------------------------------------------------------------------
    int8_net = quantize_dynamic_int8(fp32_net)
    int8_size = model_size_mb(int8_net)
    int8_latency = benchmark_latency(int8_net, state_dim, torch.device("cpu"))
    int8_agreement = evaluate_accuracy_delta(
        fp32_net, int8_net, state_dim, device, fp16=False
    )
    results["int8"] = {
        "size_mb": int8_size,
        "latency_ms": int8_latency,
        "agreement": int8_agreement,
        "size_reduction": fp32_size / max(int8_size, 1e-6),
        "speedup": fp32_latency / max(int8_latency, 1e-6),
    }
    logger.info(
        "INT8  | Size: %.3f MB (%.1fx) | Latency: %.4f ms (%.1fx) | Agreement: %.4f",
        int8_size,
        results["int8"]["size_reduction"],
        int8_latency,
        results["int8"]["speedup"],
        int8_agreement,
    )

    # Save INT8 model via TorchScript for deployment
    int8_script = torch.jit.script(int8_net)
    int8_path = output_path / "online_net_int8.pt"
    int8_script.save(str(int8_path))
    logger.info("Saved INT8 model to %s.", int8_path)

    # ------------------------------------------------------------------
    # FP16 half-precision
    # ------------------------------------------------------------------
    fp16_net = quantize_fp16(load_online_net(checkpoint_path, state_dim, action_dim, device), device)
    fp16_size = model_size_mb(fp16_net)
    fp16_latency = benchmark_latency(fp16_net, state_dim, device)
    fp16_agreement = evaluate_accuracy_delta(
        fp32_net, fp16_net, state_dim, device, fp16=True
    )
    results["fp16"] = {
        "size_mb": fp16_size,
        "latency_ms": fp16_latency,
        "agreement": fp16_agreement,
        "size_reduction": fp32_size / max(fp16_size, 1e-6),
        "speedup": fp32_latency / max(fp16_latency, 1e-6),
    }
    logger.info(
        "FP16  | Size: %.3f MB (%.1fx) | Latency: %.4f ms (%.1fx) | Agreement: %.4f",
        fp16_size,
        results["fp16"]["size_reduction"],
        fp16_latency,
        results["fp16"]["speedup"],
        fp16_agreement,
    )

    # Save FP16 state dict
    fp16_path = output_path / "online_net_fp16.pt"
    torch.save(fp16_net.state_dict(), str(fp16_path))
    logger.info("Saved FP16 model to %s.", fp16_path)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    report_path = output_path / "quantization_report.txt"
    with open(report_path, "w") as f:
        f.write("H2 Quantization Report\n")
        f.write("=" * 50 + "\n\n")
        for precision, metrics in results.items():
            f.write(f"{precision.upper()}\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
    logger.info("Report written to %s.", report_path)

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="H2 Post-Training Quantization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/checkpoint_final.pt",
        help="Path to trained FP32 checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints/quantized",
        help="Directory to save quantized models and report",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=9,
        help="Observation space dimension (NUM_HOSTS * 2 + 1)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=4,
        help="Action space dimension (n_cloud_hosts)",
    )
    args = parser.parse_args()

    results = run_quantization(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )

    print("\nQuantization complete.")
    print(f"INT8 size reduction: {results['int8']['size_reduction']:.2f}x")
    print(f"INT8 speedup:        {results['int8']['speedup']:.2f}x")
    print(f"FP16 size reduction: {results['fp16']['size_reduction']:.2f}x")
    print(f"FP16 speedup:        {results['fp16']['speedup']:.2f}x")


if __name__ == "__main__":
    main()