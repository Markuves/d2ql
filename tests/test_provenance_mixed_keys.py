"""FIX 1 (incident root cause): ``config_sha256`` must survive mixed-type keys.

The H4 sweep config mixes the integer precision key ``32`` with string keys
(``native_precision.learning_rate_overrides`` and
``capacity.max_hidden_size``). ``json.dumps(..., sort_keys=True)`` sorts the keys
by comparing them, so mixing ``int`` and ``str`` raised
``TypeError: '<' not supported between instances of 'str' and 'int'``.
Provenance is built at the END of a run (``_run_one_training``), so the crash
landed after the last episode of every run and escaped ``main()``: 50 of the 65
container restarts in the incident.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql.provenance import config_sha256

#: Same shape as configs/h4_native_precision.yaml: int keys next to str keys at
#: two different nesting levels.
MIXED_CONFIG = {
    "experiment": {"id": "h4_native_precision", "seeds": [42]},
    "agent": {"precision": "32", "hidden_size": 256},
    "native_precision": {
        "enabled": True,
        "learning_rate_overrides": {"ternary": 0.0003, 16: 0.0003, 32: 0.0005},
        "capacity": {
            "hidden_sizes": [64, 128],
            "max_hidden_size": {32: 2048, 16: 4096, "ternary": 4096},
        },
        "precisions": [{1: "nested-int-key", "precision": 32, "device": "cuda"}],
    },
}


def test_mixed_int_and_str_keys_do_not_raise():
    digest = config_sha256(MIXED_CONFIG)
    assert isinstance(digest, str)
    assert len(digest) == 16
    int(digest, 16)  # hex


def test_hash_is_deterministic_across_calls():
    assert config_sha256(MIXED_CONFIG) == config_sha256(MIXED_CONFIG)


def test_int_key_and_its_string_form_hash_the_same():
    assert config_sha256({32: "x"}) == config_sha256({"32": "x"})


def test_nested_int_keys_and_their_string_form_hash_the_same():
    assert config_sha256({"a": {32: "x"}}) == config_sha256({"a": {"32": "x"}})
    # ... also when the nested dict lives inside a list.
    assert config_sha256({"a": [{16: {"32": "x"}}]}) == config_sha256(
        {"a": [{"16": {"32": "x"}}]}
    )


def test_whole_config_is_invariant_to_key_type():
    as_str = {
        "native_precision": {
            "learning_rate_overrides": {"ternary": 0.0003, "16": 0.0003, "32": 0.0005},
            "capacity": {"max_hidden_size": {"32": 2048, "16": 4096, "ternary": 4096}},
        }
    }
    as_int = {
        "native_precision": {
            "learning_rate_overrides": {"ternary": 0.0003, 16: 0.0003, 32: 0.0005},
            "capacity": {"max_hidden_size": {32: 2048, 16: 4096, "ternary": 4096}},
        }
    }
    assert config_sha256(as_int) == config_sha256(as_str)


def test_hash_still_reacts_to_a_value_change():
    assert config_sha256({32: "x"}) != config_sha256({32: "y"})
    assert config_sha256({32: "x"}) != config_sha256({16: "x"})


def test_real_h4_config_can_be_hashed():
    """Regression guard on the exact config that killed the runs."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "h4_native_precision.yaml"
    if not cfg_path.exists():
        pytest.skip("h4 config not present")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    digest = config_sha256(cfg)
    assert len(digest) == 16
    assert digest == config_sha256(cfg)
