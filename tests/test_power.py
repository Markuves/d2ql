"""The Python power model must stay identical to the Java gateway's constants.

The reward normalisation (``E_step_max``) is derived from ``d2ql.power``; the
gateway owns the actual power calculation. If one side changes without the other
the reward silently changes scale, so this test parses the Java source.
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-agent"))

from d2ql import power  # noqa: E402

GATEWAY = (
    Path(__file__).resolve().parents[1]
    / "java-sim/src/main/java/org/d2ql/SimulationGateway.java"
)


def _java_array(source: str, name: str) -> list[float]:
    match = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", source)
    assert match, f"{name} not found in SimulationGateway.java"
    return [float(v) for v in match.group(1).split(",")]


@pytest.fixture(scope="module")
def java_source() -> str:
    if not GATEWAY.exists():
        pytest.skip("java-sim sources not present")
    return GATEWAY.read_text()


@pytest.mark.parametrize(
    "java_name, python_values",
    [
        ("HOST_IDLE_WATT", power.HOST_IDLE_WATT),
        ("HOST_MAX_WATT", power.HOST_MAX_WATT),
        ("HOST_GAMMA", power.HOST_GAMMA),
        ("HOST_PRICE_PER_WATT", power.HOST_PRICE_PER_WATT),
    ],
)
def test_power_constants_match_the_gateway(java_source, java_name, python_values):
    assert _java_array(java_source, java_name) == pytest.approx(list(python_values))
