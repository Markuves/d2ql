# ruff: noqa: N802 -- the fake mirrors the Java gateway API (camelCase).
"""FIX 4 + FIX 5 (incident): retry/backoff and uniform error handling in env.py.

When the java-sim container was mid-restart, ``setSeed`` (and every other py4j
call) failed with ``socket.gaierror`` / ``ConnectionRefusedError`` /
``Py4JNetworkError`` and the exception escaped: 13 of the 65 container deaths.
There was also no single error contract for gateway calls -- "Answer from Java
side is empty" was sometimes swallowed and sometimes fatal.

These tests inject a fake gateway that fails N times, mock the clock (no test
sleeps) and check the delay sequence, the attempt budget, that logic errors are
never retried, and that every call site behaves identically.
"""

import logging
import os
import re
import socket
import sys
from pathlib import Path

import pytest
from py4j.protocol import Py4JError, Py4JNetworkError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql import env as env_module
from d2ql.env import (
    RETRY_BACKOFF_FACTOR,
    RETRY_INITIAL_DELAY_S,
    RETRY_JITTER_FRAC,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY_S,
    CloudSimEnv,
    GatewayError,
    retry_delay,
)
from d2ql.workload import CloudletSpec

#: Value each fake gateway method returns (the env expects this shape).
_GATEWAY_RETURNS = {
    "setSeed": None,
    "clearWorkload": None,
    "addWorkloadRow": None,
    "resetEpisode": [0.0] * 12,
    "reset": [0.0] * 12,
    "step": [0.0] * 12,
    "getHostCpuUtilizations": [0.0] * 4,
    "getTotalEnergyConsumed": 0.0,
    "getMakespan": 0.0,
    "getOperationalCost": 0.0,
    "getSimulationTime": 0.0,
    "isFinished": True,
    "isEpisodeComplete": True,
    "getFinishedCloudletCount": 10,
    "isSimulationRunning": True,
}


def _empty_answer() -> Py4JNetworkError:
    # exactly the message from the incident log
    return Py4JNetworkError("Answer from Java side is empty", when="empty_response")


class FakeSim:
    """Py4J entry point stub. ``failures`` maps a method to the errors it raises."""

    def __init__(self, failures: dict | None = None):
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.calls: dict[str, int] = {}

    def _invoke(self, name: str):
        self.calls[name] = self.calls.get(name, 0) + 1
        pending = self.failures.get(name)
        if pending:
            raise pending.pop(0)
        return _GATEWAY_RETURNS[name]

    def __getattr__(self, name):
        if name not in _GATEWAY_RETURNS:
            raise AttributeError(name)
        return lambda *args, **kwargs: self._invoke(name)


@pytest.fixture
def clock(monkeypatch):
    """Record retry delays instead of sleeping, and remove the jitter."""
    delays: list[float] = []
    monkeypatch.setattr(env_module.time, "sleep", delays.append)
    monkeypatch.setattr(env_module.random, "uniform", lambda low, high: 0.0)
    return delays


def _spec(cid: int, mi: float = 1.0, pes: int = 1) -> CloudletSpec:
    return CloudletSpec(
        cloudlet_id=cid, mi=mi, num_pes=pes, memory_mb=512.0,
        submitted_at=0.0, deadline=10.0,
    )


def _env(monkeypatch, sim) -> CloudSimEnv:
    monkeypatch.setattr(CloudSimEnv, "_connect_gateway", lambda self: setattr(self, "sim", sim))
    return CloudSimEnv(config={"experiment": {"seed": 7}})


def _expected_schedule(n_retries: int) -> list[float]:
    return [
        min(RETRY_INITIAL_DELAY_S * RETRY_BACKOFF_FACTOR ** i, RETRY_MAX_DELAY_S)
        for i in range(n_retries)
    ]


# ---------------------------------------------------------------------------
# FIX 4: retry with exponential backoff
# ---------------------------------------------------------------------------


def test_setseed_retries_a_refused_connection_then_succeeds(monkeypatch, clock):
    sim = FakeSim(
        {
            "setSeed": [
                ConnectionRefusedError(111, "Connection refused"),
                ConnectionRefusedError(111, "Connection refused"),
                _empty_answer(),
            ]
        }
    )
    env = _env(monkeypatch, sim)

    env.reset(cloudlets=[_spec(0)], seed=123)  # must not raise

    assert sim.calls["setSeed"] == 4  # 3 failures + 1 success
    assert clock == _expected_schedule(3) == [1.0, 2.0, 4.0]


def test_dns_failure_is_retried(monkeypatch, clock):
    sim = FakeSim({"setSeed": [socket.gaierror(-2, "Name or service not known")]})
    env = _env(monkeypatch, sim)

    env.reset(cloudlets=[_spec(0)], seed=7)

    assert sim.calls["setSeed"] == 2
    assert clock == [1.0]


def test_backoff_schedule_is_exponential_and_capped(monkeypatch):
    monkeypatch.setattr(env_module.random, "uniform", lambda low, high: 0.0)
    schedule = [retry_delay(attempt) for attempt in range(1, 11)]
    assert schedule[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert schedule[5:] == [RETRY_MAX_DELAY_S] * 5
    assert all(delay <= RETRY_MAX_DELAY_S for delay in schedule)


def test_backoff_has_jitter_within_bounds(monkeypatch):
    monkeypatch.setattr(env_module.random, "uniform", lambda low, high: high)
    assert retry_delay(1) == pytest.approx(RETRY_INITIAL_DELAY_S * (1 + RETRY_JITTER_FRAC))
    monkeypatch.setattr(env_module.random, "uniform", lambda low, high: low)
    assert retry_delay(1) == pytest.approx(RETRY_INITIAL_DELAY_S * (1 - RETRY_JITTER_FRAC))


def test_each_retry_is_logged_at_warning(monkeypatch, clock, caplog):
    caplog.set_level(logging.DEBUG)
    sim = FakeSim({"setSeed": [_empty_answer(), _empty_answer()]})
    env = _env(monkeypatch, sim)

    env.reset(cloudlets=[_spec(0)], seed=7)

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "setSeed" in r.getMessage()
    ]
    assert len(warnings) == 2
    assert "attempt 1/" in warnings[0].getMessage()
    assert "attempt 2/" in warnings[1].getMessage()


def test_retries_stop_after_the_attempt_budget(monkeypatch, clock):
    sim = FakeSim({"setSeed": [_empty_answer() for _ in range(50)]})
    env = _env(monkeypatch, sim)

    with pytest.raises(GatewayError):
        env.reset(cloudlets=[_spec(0), _spec(1)], seed=7)

    assert sim.calls["setSeed"] == RETRY_MAX_ATTEMPTS
    assert len(clock) == RETRY_MAX_ATTEMPTS - 1
    assert clock == _expected_schedule(RETRY_MAX_ATTEMPTS - 1)


def test_logic_errors_are_not_retried(monkeypatch, clock):
    sim = FakeSim({"setSeed": [ValueError("bad seed"), ValueError("bad seed")]})
    env = _env(monkeypatch, sim)

    with pytest.raises(ValueError):
        env.reset(cloudlets=[_spec(0)], seed=7)

    assert sim.calls["setSeed"] == 1
    assert clock == []


def test_non_network_py4j_errors_are_not_retried(monkeypatch, clock):
    """A Java-side logic error must surface at once, not after 10 attempts."""
    sim = FakeSim({"step": [Py4JError("no such method: step")]})
    env = _env(monkeypatch, sim)

    with pytest.raises(Py4JError):
        env.step(0)

    assert sim.calls["step"] == 1
    assert clock == []


def test_gateway_connection_is_retried(monkeypatch, clock):
    """java-sim mid-restart: JavaGateway() itself raises until the JVM is up."""
    attempts: list[int] = []
    sim = FakeSim()

    class FakeGateway:
        def __init__(self, gateway_parameters=None):
            attempts.append(1)
            if len(attempts) <= 2:
                raise ConnectionRefusedError(111, "Connection refused")
            self.entry_point = sim

    monkeypatch.setattr("py4j.java_gateway.JavaGateway", FakeGateway)

    env = CloudSimEnv(config={"py4j": {"host": "java-sim", "port": 25333}})

    assert env.sim is sim
    assert len(attempts) == 3
    assert clock == _expected_schedule(2) == [1.0, 2.0]


def test_failed_connection_raises_the_gateway_error(monkeypatch, clock):
    class AlwaysRefused:
        def __init__(self, gateway_parameters=None):
            raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr("py4j.java_gateway.JavaGateway", AlwaysRefused)

    with pytest.raises(GatewayError) as excinfo:
        CloudSimEnv(config={"py4j": {"host": "java-sim", "port": 25333}})

    assert "connect" in excinfo.value.call_site
    assert len(clock) == RETRY_MAX_ATTEMPTS - 1


def test_reset_survives_a_gateway_that_comes_back(monkeypatch, clock):
    """End-to-end: the first episode hits a restarting gateway, the next is clean."""
    sim = FakeSim({"setSeed": [ConnectionRefusedError(111, "refused")]})
    env = _env(monkeypatch, sim)

    env.reset(cloudlets=[_spec(0)], seed=1)  # retried, then succeeded
    env.reset(cloudlets=[_spec(0)], seed=2)

    assert sim.calls["setSeed"] == 3  # 1 failed + retry + second episode
    assert clock == [1.0]


# ---------------------------------------------------------------------------
# FIX 5: every call site goes through the same path, with one error type
# ---------------------------------------------------------------------------

def _cloudlets():
    return [_spec(0), _spec(1)]


#: gateway method -> how to trigger it through the public env API
CALL_SITE_TRIGGERS = {
    "setSeed": lambda env: env.reset(cloudlets=_cloudlets(), seed=1),
    "clearWorkload": lambda env: env.reset(cloudlets=_cloudlets(), seed=1),
    "addWorkloadRow": lambda env: env.reset(cloudlets=_cloudlets(), seed=1),
    "resetEpisode": lambda env: env.reset(cloudlets=_cloudlets(), seed=1),
    "reset": lambda env: env.reset(),
    "step": lambda env: env.step(0),
    "getHostCpuUtilizations": lambda env: env.step(0),
    "getTotalEnergyConsumed": lambda env: env.step(0),
    "getMakespan": lambda env: env.step(0),
    "getOperationalCost": lambda env: env.step(0),
    "getSimulationTime": lambda env: env.step(0),
    "getFinishedCloudletCount": lambda env: env.step(0),
    "isEpisodeComplete": lambda env: env.step(0),
    "isSimulationRunning": lambda env: env.step(0),
    # only reached by gateways too old to expose isEpisodeComplete()
    "isFinished": lambda env: (
        setattr(env, "_supports_episode_complete", False),
        env.step(0),
    ),
}


def test_the_audit_covers_every_py4j_call_site_in_env_py():
    """No direct py4j call may bypass the retry/error path."""
    source = Path(env_module.__file__).read_text()
    assert not re.findall(r"self\.sim\.\w+\(", source), (
        "a py4j call bypasses _gateway_call(): "
        f"{re.findall(r'self.sim.*', source)}"
    )
    routed = set(re.findall(r'_gateway_call\(\s*"(\w+)"', source))
    assert routed == set(CALL_SITE_TRIGGERS), (
        f"call sites not covered by the audit: {routed ^ set(CALL_SITE_TRIGGERS)}"
    )


@pytest.mark.parametrize("site", sorted(CALL_SITE_TRIGGERS))
def test_every_call_site_has_the_same_retry_policy(monkeypatch, clock, site):
    sim = FakeSim({site: [_empty_answer() for _ in range(50)]})
    env = _env(monkeypatch, sim)

    with pytest.raises(GatewayError) as excinfo:
        CALL_SITE_TRIGGERS[site](env)

    error = excinfo.value
    # Same exception type everywhere, carrying the failing call site.
    assert type(error) is GatewayError
    assert error.call_site == site
    # Same attempt budget and same backoff schedule everywhere.
    assert sim.calls[site] == RETRY_MAX_ATTEMPTS
    assert clock == _expected_schedule(RETRY_MAX_ATTEMPTS - 1)


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: ConnectionRefusedError(111, "Connection refused"),
        lambda: socket.gaierror(-2, "Name or service not known"),
        _empty_answer,
    ],
    ids=["ConnectionRefusedError", "gaierror", "Py4JNetworkError"],
)
def test_every_transient_error_kind_is_retried_identically(monkeypatch, clock, make_error):
    sim = FakeSim({"setSeed": [make_error(), make_error()]})
    env = _env(monkeypatch, sim)

    env.reset(cloudlets=[_spec(0)], seed=1)

    assert sim.calls["setSeed"] == 3
    assert clock == _expected_schedule(2)


@pytest.mark.parametrize("site", ["setSeed", "step", "getMakespan"])
def test_persistent_emptiness_surfaces_with_episode_context(monkeypatch, clock, site):
    """A wedged gateway must not leak a raw py4j traceback without context."""
    sim = FakeSim({site: [_empty_answer() for _ in range(50)]})
    env = _env(monkeypatch, sim)
    if site != "setSeed":
        CALL_SITE_TRIGGERS["setSeed"](env)  # a healthy reset first

    with pytest.raises(GatewayError) as excinfo:
        CALL_SITE_TRIGGERS[site](env)

    error = excinfo.value
    assert error.context["episode"] >= 1
    if site == "setSeed":
        assert error.context["cloudlets"] == 2
    message = str(error)
    assert site in message
    assert "episode=" in message
    assert "cloudlets=" in message
    # The raw py4j exception stays attached as the cause.
    assert isinstance(error.__cause__, Py4JNetworkError)


def test_old_gateway_fallbacks_still_work(monkeypatch, clock):
    """Uniform handling must not break the missing-method fallbacks (SIM-08)."""
    sim = FakeSim({"isEpisodeComplete": [Py4JError("java.lang.NoSuchMethodError")]})
    env = _env(monkeypatch, sim)

    _, _, _, _, info = env.step(0)

    # The capability probe failed once (a logic error, so not retried) and the
    # env fell back to isFinished() instead of dying.
    assert sim.calls["isEpisodeComplete"] == 1
    assert sim.calls["isFinished"] == 1
    assert info["finished_cloudlets"] == 10
    assert env._supports_episode_complete is False


def test_env_close_tolerates_a_dead_gateway(monkeypatch, clock):
    sim = FakeSim()

    class DeadClient:
        def close(self):
            raise ConnectionRefusedError("already gone")

    env = _env(monkeypatch, sim)
    env.gateway = DeadClient()
    env.close()  # best effort, must not raise
    assert env.gateway is None
