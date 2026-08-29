import os
import sys

import torch
import torch.nn.functional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from d2ql.precision import (
    NativeBitLinear,
    effective_capacity_bits,
    lowbit_real_matmul,
    model_flops,
    model_macs,
)

# ---------------------------------------------------------------------------
# B2: compute-cost normalization
# ---------------------------------------------------------------------------

def test_model_macs_shape():
    # state(9) -> hidden(256); hidden -> hidden(256); hidden -> action(4)
    macs = model_macs(9, 256, 2, 4)
    expected = 9 * 256 + 256 * 256 + 256 * 4
    assert macs == expected


def test_model_flops_is_twice_macs():
    assert model_flops(9, 256, 2, 4) == 2.0 * model_macs(9, 256, 2, 4)


def test_effective_capacity_bits():
    assert effective_capacity_bits(1000, 4) == 4000.0
    assert effective_capacity_bits(1000, 1.58) == 1580.0


# ---------------------------------------------------------------------------
# B1: real low-bit int8 matmul kernel
# ---------------------------------------------------------------------------

def test_lowbit_real_matmul_approximates_linear():
    torch.manual_seed(0)
    batch, in_f, out_f = 8, 16, 4
    x = torch.rand(batch, in_f) * 0.5
    weight = (torch.rand(out_f, in_f) - 0.5) * 2
    bias = torch.rand(out_f) * 0.1

    y_real = lowbit_real_matmul(x, weight, bias)
    y_ref = torch.nn.functional.linear(x, weight, bias)

    assert y_real.shape == y_ref.shape
    # int8 symmetric quantization introduces small relative error.
    torch.testing.assert_close(y_real, y_ref, rtol=0.1, atol=0.1)


def test_lowbit_real_matmul_no_bias():
    torch.manual_seed(1)
    x = torch.rand(5, 12)
    weight = torch.rand(4, 12) - 0.5
    y_real = lowbit_real_matmul(x, weight, None)
    y_ref = torch.nn.functional.linear(x, weight)
    assert y_real.shape == (5, 4)
    torch.testing.assert_close(y_real, y_ref, rtol=0.1, atol=0.1)


def test_native_bit_linear_deploy_matches_ste_approximately():
    torch.manual_seed(2)
    layer = NativeBitLinear(16, 8, precision="8", quantize_activations=True)
    x = torch.rand(4, 16)
    out_ste = layer(x)          # fake-quant STE path (training)
    layer.deploy = True
    out_deploy = layer(x)       # real int8 kernel (B1, deploy)
    layer.deploy = False
    assert out_ste.shape == out_deploy.shape
    torch.testing.assert_close(out_deploy, out_ste, rtol=0.2, atol=0.2)


def test_set_deploy_toggles_all_grid_layers():
    layer1 = NativeBitLinear(16, 16, precision="8", quantize_activations=True)
    layer2 = NativeBitLinear(16, 4, precision="ternary", quantize_activations=True)
    assert not layer1.deploy and not layer2.deploy
    for layer in (layer1, layer2):
        layer.deploy = True
    assert layer1.deploy and layer2.deploy
