# ruff: noqa: N802 -- the fake mirrors the Java gateway API (camelCase).
"""The env must send the episode window to the gateway in dispatch (LPT) order.

SIM-04: the queue used to be computed and then ignored. These tests use a stub
Py4J entry point, so they run without a gateway.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql import env as env_module
from d2ql.env import OBS_DIM, CloudSimEnv
from d2ql.workload import CloudletSpec


class FakeSim:
    """Records what the env pushes to the gateway."""

    def __init__(self):
        self.rows = []
        self.seed = None
        self.steps = []
        self.energy = 0.0

    def setSeed(self, seed):
        self.seed = seed

    def clearWorkload(self):
        self.rows = []

    def addWorkloadRow(self, submitted_at, deadline, mi, num_pes):
        self.rows.append((float(mi), int(num_pes)))

    def resetEpisode(self):
        return [0.0] * 12  # gateway still emits 12 values

    def reset(self):
        return [0.0] * 12

    def step(self, host_index):
        self.steps.append(int(host_index))
        return [0.0] * 12

    def getHostCpuUtilizations(self):
        return [0.0] * 4

    def getTotalEnergyConsumed(self):
        return self.energy

    def getMakespan(self):
        return 0.0

    def getOperationalCost(self):
        return 0.0

    def getSimulationTime(self):
        return 0.0

    def isFinished(self):
        return True

    def isEpisodeComplete(self):
        return True


@pytest.fixture
def stub_env(monkeypatch):
    monkeypatch.setattr(CloudSimEnv, "_connect_gateway", lambda self: setattr(self, "sim", FakeSim()))
    return CloudSimEnv(config={"experiment": {"seed": 7}})


def _spec(cid: int, mi: float, pes: int) -> CloudletSpec:
    return CloudletSpec(
        cloudlet_id=cid, mi=mi, num_pes=pes, memory_mb=512.0,
        submitted_at=0.0, deadline=10.0,
    )


def test_reset_sends_cloudlets_in_lpt_order(stub_env):
    # work per PE: id1 = 400, id2 = 200, id0 = 50 -> LPT order 1, 2, 0
    specs = [_spec(0, 50.0, 1), _spec(1, 400.0, 1), _spec(2, 400.0, 2)]
    stub_env.reset(cloudlets=specs)
    assert [mi for mi, _ in stub_env.sim.rows] == [400.0, 400.0, 50.0]
    assert [pes for _, pes in stub_env.sim.rows] == [1, 2, 1]


def test_reset_seeds_the_gateway(stub_env):
    stub_env.reset(cloudlets=[_spec(0, 1.0, 1)], seed=123)
    assert stub_env.sim.seed == 123


def test_observation_drops_the_constant_deadline_feature(stub_env):
    obs, _ = stub_env.reset(cloudlets=[_spec(0, 1.0, 1)])
    assert obs.shape == (OBS_DIM,)
    assert OBS_DIM == 11
    assert env_module.N_HOSTS == 4


def test_step_returns_energy_delta_and_no_sla(stub_env):
    _, _, _, _, info = stub_env.step(1)
    assert "energy_delta" in info
    assert "sla_violations" not in info
    assert stub_env.sim.steps == [1]


def test_step_returns_the_reward_from_the_energy_delta(stub_env):
    # First step: no energy consumed -> reward 0.
    _, reward0, _, _, _ = stub_env.step(0)
    assert reward0 == pytest.approx(0.0)

    # Second step: a full E_step_max of energy -> reward -1.
    stub_env.sim.energy = stub_env.reward_fn.e_step_max
    _, reward1, _, _, info = stub_env.step(0)
    assert info["energy_delta"] == pytest.approx(stub_env.reward_fn.e_step_max)
    assert reward1 == pytest.approx(-1.0)


def test_env_reward_matches_the_reward_manager(stub_env):
    from d2ql.reward import RewardManager

    assert isinstance(stub_env.reward_fn, RewardManager)
    assert stub_env.reward_fn.e_step_max > 0
