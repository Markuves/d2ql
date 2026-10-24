"""Provenance block: every persisted row must say which code/config produced it."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql.provenance import SCHEMA_VERSION, config_sha256, git_sha, provenance, run_timestamp


def test_provenance_has_the_required_keys():
    block = provenance({"agent": {"precision": "32", "hidden_size": 64}})
    assert block["schema_version"] == SCHEMA_VERSION == 2
    assert block["git_sha"]
    assert len(block["config_sha256"]) == 16
    assert block["run_timestamp_utc"].endswith("+00:00")


def test_extra_fields_are_merged():
    block = provenance({"a": 1}, obs_dim=11, dispatch_order="lpt_work_per_pe")
    assert block["obs_dim"] == 11
    assert block["dispatch_order"] == "lpt_work_per_pe"


def test_config_hash_is_stable_and_sensitive():
    cfg = {"agent": {"precision": "32"}, "training": {"n_episodes": 4}}
    assert config_sha256(cfg) == config_sha256(dict(cfg))
    assert config_sha256(cfg) != config_sha256({"agent": {"precision": "16"}})


def test_git_sha_is_never_empty(monkeypatch):
    monkeypatch.setenv("D2QL_GIT_SHA", "deadbee")
    assert git_sha() == "deadbee"


def test_run_timestamp_is_utc_iso():
    stamp = run_timestamp()
    assert "T" in stamp and stamp.endswith("+00:00")
