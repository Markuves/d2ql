"""FIX 6 (incident investigation): java-sim diagnostics + log retention + probe.

Two mid-training ``Py4JNetworkError`` crashes suggest the java-sim JVM died under
load (Dockerfile.java sets ``-XX:+ExitOnOutOfMemoryError``), but it could not be
confirmed because the java-sim logs were not captured. These tests check the
configuration that makes the next occurrence diagnosable:

* java-sim emits GC logs and a heap dump on OOM into a **mounted** volume.
* python-agent keeps more Docker log history (100 MB x 5 instead of 10 MB x 3).
* the java-sim healthcheck really probes the Py4J port 25333 instead of trusting
  that the container process is alive.

The Docker-backed assertions are skipped when no daemon/image is available;
the configuration assertions always run.
"""

import json
import os
import re
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSE = os.path.join(REPO_ROOT, "docker-compose.yml")
DOCKERFILE_JAVA = os.path.join(REPO_ROOT, "Dockerfile.java")
JAVA_IMAGE = "d2ql-java-sim:latest"
JAVA_CONTAINER = "java-sim"
PORT = 25333


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _has_docker_cli() -> bool:
    return shutil.which("docker") is not None


def _docker_daemon_ok() -> bool:
    if not _has_docker_cli():
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=30
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _running_container(name: str) -> bool:
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True, text=True, timeout=30,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


def _image_present(tag: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", tag], capture_output=True, timeout=30
    ).returncode == 0


needs_docker = pytest.mark.skipif(
    not _docker_daemon_ok(), reason="docker daemon not available"
)


@pytest.fixture(scope="module")
def compose_config() -> dict:
    """`docker compose config` resolved to JSON (also validates the file)."""
    if not _has_docker_cli():
        pytest.skip("docker CLI not available")
    proc = subprocess.run(
        ["docker", "compose", "-f", COMPOSE, "config", "--format", "json"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"docker compose config failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def healthcheck_cmd(compose_config) -> str:
    test = compose_config["services"]["java-sim"]["healthcheck"]["test"]
    parts = test if isinstance(test, list) else [test]
    if parts and parts[0] in ("CMD-SHELL", "CMD"):
        parts = parts[1:]  # drop the Compose/exec prefix, keep the shell command
    return " ".join(parts)


def _port_probe(healthcheck_cmd: str) -> str:
    """The TCP connect clause of the healthcheck (no /proc, no process checks)."""
    match = re.search(
        r"timeout \d+ bash -c '(exec \d+<>/dev/tcp/[\w.]+/\d+|cat < /dev/null > /dev/tcp/[\w.]+/\d+)'",
        healthcheck_cmd,
    )
    assert match, f"no TCP connect probe found in healthcheck: {healthcheck_cmd!r}"
    clause = match.group(0)
    assert str(PORT) in clause
    return clause


def _dockerfile_entrypoint_flags() -> list[str]:
    """JVM flags from the Dockerfile ENTRYPOINT (the ``-X``/``--`` options)."""
    text = open(DOCKERFILE_JAVA).read()
    match = re.search(r"ENTRYPOINT \[(.*?)\]", text, re.S)
    assert match, "Dockerfile.java has no ENTRYPOINT array"
    items = re.findall(r'"([^"]*)"', match.group(1))
    return [arg for arg in items if arg.startswith(("-X", "--"))]


# ---------------------------------------------------------------------------
# java-sim keeps evidence of a JVM death
# ---------------------------------------------------------------------------


def test_dockerfile_emits_gc_logs_to_the_mounted_path():
    text = open(DOCKERFILE_JAVA).read()
    assert "-Xlog:gc*:file=/app/logs/gc.log" in text, "GC logging is not enabled"
    # The path must be the mounted volume, not the container's ephemeral layer.
    assert "java-logs:/app/logs" in open(COMPOSE).read()


def test_dockerfile_writes_a_heap_dump_on_oom():
    text = open(DOCKERFILE_JAVA).read()
    assert "-XX:+HeapDumpOnOutOfMemoryError" in text
    assert "-XX:HeapDumpPath=/app/logs/" in text


def test_dockerfile_prepares_the_log_directory_for_the_non_root_user():
    """The JVM exits if it cannot open the -Xlog file, so /app/logs must be
    writable by uid 999 (d2ql) before the ENTRYPOINT runs."""
    text = open(DOCKERFILE_JAVA).read()
    assert re.search(r"mkdir -p /app/logs.*chown d2ql:d2ql /app/logs", text), (
        "Dockerfile.java must create /app/logs owned by d2ql"
    )


def test_compose_mounts_the_log_volume_on_java_sim(compose_config):
    mounts = compose_config["services"]["java-sim"]["volumes"]
    targets = {m.get("target") for m in mounts if isinstance(m, dict)}
    assert "/app/logs" in targets, mounts


def test_simulation_flags_are_unchanged():
    """FIX 6 must not touch simulation behaviour (JVM/memory flags stay)."""
    flags = _dockerfile_entrypoint_flags()
    for expected in (
        "-XX:+UseContainerSupport",
        "-XX:InitialRAMPercentage=50.0",
        "-XX:MaxRAMPercentage=75.0",
        "-XX:+UseG1GC",
        "-XX:+ExitOnOutOfMemoryError",
        "--enable-native-access=ALL-UNNAMED",
    ):
        assert expected in flags, f"{expected} disappeared from the ENTRYPOINT"


# ---------------------------------------------------------------------------
# python-agent log retention
# ---------------------------------------------------------------------------


def test_python_agent_keeps_more_log_history(compose_config):
    options = compose_config["services"]["python-agent"]["logging"]["options"]
    assert options["max-size"] == "100m"
    assert options["max-file"] == "5"


# ---------------------------------------------------------------------------
# healthcheck: the port, not the process
# ---------------------------------------------------------------------------


def test_healthcheck_probes_the_gateway_port(healthcheck_cmd):
    assert str(PORT) in healthcheck_cmd
    assert re.search(r"timeout \d+ bash -c '(exec \d+<>/dev/tcp|cat < /dev/null > /dev/tcp)", healthcheck_cmd), (
        f"healthcheck does not open a TCP connection: {healthcheck_cmd!r}"
    )
    # A 1s connect timeout reports a busy-but-alive JVM as unhealthy.
    timeout_s = int(re.search(r"timeout (\d+) bash", healthcheck_cmd).group(1))
    assert timeout_s >= 3, "healthcheck connect timeout is too tight for a loaded JVM"


@needs_docker
def test_healthcheck_passes_on_the_running_java_sim(healthcheck_cmd):
    if not _running_container(JAVA_CONTAINER):
        pytest.skip(f"container {JAVA_CONTAINER} is not running")
    proc = subprocess.run(
        ["docker", "exec", JAVA_CONTAINER, "sh", "-c", healthcheck_cmd],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@needs_docker
def test_port_probe_fails_when_nothing_listens(healthcheck_cmd):
    """A java-sim that is not serving the gateway must not look healthy."""
    if not _image_present(JAVA_IMAGE):
        pytest.skip(f"image {JAVA_IMAGE} not built")
    probe = _port_probe(healthcheck_cmd)
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", JAVA_IMAGE, "-c", probe],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0, (
        "the port probe reported success with nothing listening on 25333"
    )


@needs_docker
def test_port_probe_passes_while_the_gateway_listens(healthcheck_cmd):
    if not _running_container(JAVA_CONTAINER):
        pytest.skip(f"container {JAVA_CONTAINER} is not running")
    probe = _port_probe(healthcheck_cmd)
    proc = subprocess.run(
        ["docker", "exec", JAVA_CONTAINER, "sh", "-c", probe],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@needs_docker
def test_jvm_accepts_the_diagnostic_flags_and_writes_gc_log():
    """The exact ENTRYPOINT flags must not stop the JVM from starting.

    ``-Xlog`` is fatal when the log file cannot be opened, so a wrong path in the
    new flags would take java-sim down instead of instrumenting it.
    """
    if not _image_present(JAVA_IMAGE):
        pytest.skip(f"image {JAVA_IMAGE} not built")
    flags = [
        arg.replace("/app/logs", "/tmp/d2ql-logs") for arg in _dockerfile_entrypoint_flags()
        if not arg.startswith("-jar")
    ]
    script = "mkdir -p /tmp/d2ql-logs && java " + " ".join(flags) + " -version"
    script += " && ls -l /tmp/d2ql-logs/gc.log"
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", JAVA_IMAGE, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "gc.log" in proc.stdout
