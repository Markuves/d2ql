"""Tests for the H4 sweep plan: depth vector and precision x width expansion."""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python-agent"))

from main import _build_native_plan, _experiment_seeds, _resolve_layer_counts

# ---------------------------------------------------------------------------
# n_hidden_layers as an int or a vector
# ---------------------------------------------------------------------------

def test_resolve_layer_counts_accepts_int_and_list():
    assert _resolve_layer_counts({"n_hidden_layers": 2}, {}) == [2]
    # A list is de-duplicated and sorted.
    assert _resolve_layer_counts({"n_hidden_layers": [2, 1, 2]}, {}) == [1, 2]


def test_resolve_layer_counts_falls_back_to_agent_config():
    assert _resolve_layer_counts({}, {"n_hidden_layers": 3}) == [3]


def test_resolve_layer_counts_rejects_empty_or_invalid():
    with pytest.raises(ValueError):
        _resolve_layer_counts({"n_hidden_layers": []}, {})
    with pytest.raises(ValueError):
        _resolve_layer_counts({"n_hidden_layers": [0, -1]}, {})


# ---------------------------------------------------------------------------
# Plan expansion
# ---------------------------------------------------------------------------

def test_build_native_plan_expands_depth_and_respects_cap():
    precisions = [
        {"precision": 32, "device": "cuda"},
        {"precision": "ternary", "device": "cpu"},
    ]
    plan = _build_native_plan(
        precisions, [], [256, 4096], [1, 2], {"32": 2048, "ternary": 4096}
    )
    # fp32 is capped at 2048, so width 4096 is dropped for it; ternary takes both.
    assert ("32", "cuda", 256, 1) in plan
    assert ("32", "cuda", 256, 2) in plan
    assert ("32", "cuda", 4096, 1) not in plan
    assert ("ternary", "cpu", 4096, 2) in plan
    assert len(plan) == 2 + 4  # fp32: 1 width; ternary: 2 widths; both x 2 depths


def test_build_native_plan_defaults_device_to_auto():
    plan = _build_native_plan([{"precision": "16"}], [], [256], [1], {})
    assert plan == [("16", "auto", 256, 1)]


def test_build_native_plan_bits_list_path():
    plan = _build_native_plan([], [32, 16], [256], [1, 2], {})
    assert plan == [
        ("32", "auto", 256, 1),
        ("32", "auto", 256, 2),
        ("16", "auto", 256, 1),
        ("16", "auto", 256, 2),
    ]


# ---------------------------------------------------------------------------
# Regression guard on the real H4 config
# ---------------------------------------------------------------------------

def test_h4_config_expands_to_expected_plan():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "h4_native_precision.yaml"
    if not cfg_path.exists():
        pytest.skip("h4 config not present")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    native = cfg["native_precision"]
    cap = native["capacity"]
    layers = _resolve_layer_counts(cap, cfg.get("agent", {}))
    plan = _build_native_plan(
        native["precisions"],
        native.get("bits") or [],
        [int(h) for h in cap["hidden_sizes"]],
        layers,
        cap.get("max_hidden_size") or {},
    )
    seeds = _experiment_seeds(cfg)

    assert layers == [1, 2]
    assert seeds == [42, 43, 44]
    # Widths per precision: fp32 (capped 2048) x2 -> 4 each; 16/bf16/ternary -> 5.
    assert len(plan) == (4 + 5 + 5 + 4 + 5) * 2
    assert len(plan) * len(seeds) == 138
