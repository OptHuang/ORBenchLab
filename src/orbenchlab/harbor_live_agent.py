"""Real arm executor for the live-intervention study.

Composes the pieces P0-B provides into one arm: build/start a NO-NETWORK task
container from the task's ``environment`` image, start the host-side credential
relay with a per-arm scoped token, write an MCP config pointing the agent's
only tools at the container proxy, run the live-intervention session with the
Claude CLI on the host, then grade the resulting container with the SEPARATE
frozen verifier to obtain reward + CTRF.  The two independent pieces of
evidence — the interrupt/hint journal and the verifier reward/CTRF — are
returned joined by container identity for the study to validate.

The container/verifier primitives are an injected ``ContainerBackend`` so the
composition (no-network enforcement, credential isolation, tool routing,
verifier join, guaranteed teardown) is unit-tested with a fake, and the same
executor runs Docker on the execution host.  The provider credential lives only
in the host relay: it is never placed in the container env, the agent argv/env,
``/proc``, config, logs, ATIF, or any receipt.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from . import harbor_live_intervention as hli
from .core.errors import ORBenchError


class LiveAgentError(ORBenchError):
    exit_code = 8


@dataclass
class ExecOutcome:
    rc: int
    stdout: str
    stderr: str


class ContainerBackend(Protocol):
    """Container primitives the arm executor needs; docker-backed in production."""

    def build_image(self, *, context_dir: Path, tag: str) -> str: ...

    def run_no_network(self, *, image: str, name: str, env: Mapping[str, str]) -> str:
        """Start a detached container with NO network and return its id."""

    def exec(self, *, container_id: str, argv: Sequence[str], stdin: bytes = b"") -> ExecOutcome: ...

    def read_text(self, *, container_id: str, path: str) -> str | None: ...

    def stop(self, *, container_id: str) -> None: ...


# ---------------------------------------------------------------------------


def _load_reward_and_ctrf(backend: ContainerBackend, container_id: str) -> tuple[float | None, dict | None]:
    """Read the frozen verifier's reward.txt and ctrf.json from the container."""

    reward: float | None = None
    raw_reward = backend.read_text(container_id=container_id, path="/logs/verifier/reward.txt")
    if raw_reward is not None:
        try:
            reward = float(raw_reward.strip())
        except ValueError:
            reward = None
    ctrf: dict | None = None
    raw_ctrf = backend.read_text(container_id=container_id, path="/logs/verifier/ctrf.json")
    if raw_ctrf is not None:
        try:
            parsed = json.loads(raw_ctrf)
            if isinstance(parsed, dict):
                ctrf = parsed
        except json.JSONDecodeError:
            ctrf = None
    return reward, ctrf


def _digest_text(text: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def build_arm_executor(
    *,
    task_dir: str | Path,
    backend: ContainerBackend,
    relay_factory,
    claude_executable: str | Path,
    proxy_argv_prefix: Sequence[str],
    verifier_argv: Sequence[str],
    model: str,
    max_turns: int,
    max_budget_usd: float,
    secret_values: Sequence[str] = (),
    make_policy=None,
    session_runner=None,
):
    """Return an ``ArmExecutor`` for :func:`harbor_intervention_study`.

    ``relay_factory()`` yields a context manager producing a running host relay
    exposing ``issue_token(scope)`` and ``relay_env(token)`` (the P0-A relay).
    ``proxy_argv_prefix`` launches the container MCP proxy for a container id
    (e.g. ``[python, -m, orbenchlab.harbor_container_proxy]``); the executor
    appends the container id.  ``verifier_argv`` runs the frozen verifier inside
    the container (e.g. ``["/bin/sh", "/tests/test.sh"]``).  ``make_policy``
    maps a level to an :class:`~harbor_live_intervention.InterventionPolicy`.
    """

    task_root = Path(task_dir).resolve()
    env_context = task_root / "environment"
    if not (env_context / "Dockerfile").is_file():
        raise LiveAgentError("task environment/Dockerfile is required for the live container")

    def _policy(level: str) -> hli.InterventionPolicy:
        if make_policy is not None:
            return make_policy(level)
        if level == "baseline":
            return hli.InterventionPolicy(level="baseline")
        marker = f"LIVE_HINT_{level}"
        return hli.InterventionPolicy(
            level=level,
            hint_text=f"{marker}: reconsider the approach and correct the solution.",
            hint_marker=marker,
            trigger=hli.Trigger("tool-use"),
        )

    def executor(level: str, repeat: int, intervention_id: str, journal_dir: Path) -> dict[str, Any]:
        journal_dir = Path(journal_dir)
        journal_dir.mkdir(parents=True, exist_ok=True)
        image = backend.build_image(context_dir=env_context, tag=f"orbench-live-{intervention_id[:16]}")
        container_name = f"orbench-live-{intervention_id[:24]}"
        # The container gets NO provider credential and NO network: the agent can
        # neither read the secret nor egress from its tools.
        container_id = backend.run_no_network(image=image, name=container_name, env={})
        budget = 0.0
        try:
            with relay_factory() as relay:
                token = relay.issue_token({"intervention_id": intervention_id, "level": level})
                relay_env = relay.relay_env(token)
                mcp_config_path = journal_dir / "mcp.json"
                proxy_argv = [str(part) for part in proxy_argv_prefix] + [container_id]
                mcp_config = {"mcpServers": {hli_proxy_name(): {"command": proxy_argv[0], "args": proxy_argv[1:]}}}
                mcp_config_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
                argv, child_env = hli.build_claude_live_command(
                    claude_executable=claude_executable,
                    model=model,
                    max_budget_usd=max_budget_usd,
                    max_turns=max_turns,
                    mcp_config_path=mcp_config_path,
                    proxy_server_name=hli_proxy_name(),
                    proxy_tool_names=["bash", "read_file", "write_file"],
                    relay_route=relay_env["ANTHROPIC_BASE_URL"],
                    relay_token=relay_env["ANTHROPIC_AUTH_TOKEN"],
                    claude_config_dir=journal_dir / "claude-config",
                )
                (journal_dir / "claude-config").mkdir(exist_ok=True)
                instruction = _read_instruction(task_root)
                run_session = session_runner if session_runner is not None else hli.run_live_intervention
                journal = run_session(
                    argv=argv,
                    env=child_env,
                    cwd=journal_dir,
                    policy=_policy(level),
                    initial_prompt=instruction,
                    intervention_id=intervention_id,
                    journal_dir=journal_dir,
                    timeout_sec=max(60.0, max_turns * 30.0),
                    secret_values=list(secret_values),
                    harbor_identity={"container_id": container_id, "task_dir": str(task_root), "level": level, "repeat": repeat},
                    budget_max_usd=max_budget_usd,
                )
                budget = max_budget_usd
            # Grade with the SEPARATE frozen verifier inside the same container.
            verify = backend.exec(container_id=container_id, argv=list(verifier_argv))
            reward, ctrf = _load_reward_and_ctrf(backend, container_id)
        finally:
            backend.stop(container_id=container_id)

        return {
            "journal": journal,
            "reward": reward,
            "ctrf": ctrf,
            "reward_digest": _digest_text(str(reward)) if reward is not None else None,
            "ctrf_digest": _digest_text(json.dumps(ctrf, sort_keys=True)) if ctrf is not None else None,
            "verifier_container_id": container_id,
            "verifier_exit_code": verify.rc,
            "budget_usd": budget,
        }

    return executor


def hli_proxy_name() -> str:
    from . import harbor_container_proxy

    return harbor_container_proxy.SERVER_NAME


def _read_instruction(task_root: Path) -> str:
    for name in ("instruction.md", "README.md"):
        candidate = task_root / name
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise LiveAgentError("task has no instruction.md/README.md")


class DockerBackend:
    """Production ``ContainerBackend`` backed by the Docker CLI.

    Containers are started with ``--network none`` and no provider credential in
    their environment, so an agent's tools cannot egress or read the secret.
    """

    def __init__(self, *, docker_bin: str = "docker", exec_timeout_sec: float = 600.0) -> None:
        self._docker = docker_bin
        self._timeout = exec_timeout_sec

    def _run(self, args: Sequence[str], *, stdin: bytes = b"", timeout: float | None = None) -> ExecOutcome:
        import subprocess

        proc = subprocess.run(
            [self._docker, *[str(a) for a in args]],
            input=stdin,
            capture_output=True,
            timeout=timeout if timeout is not None else self._timeout,
        )
        return ExecOutcome(
            rc=proc.returncode,
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
        )

    def build_image(self, *, context_dir: Path, tag: str) -> str:
        out = self._run(["build", "-t", tag, str(context_dir)], timeout=1800)
        if out.rc != 0:
            raise LiveAgentError(f"docker build failed: {out.stderr[-500:]}")
        return tag

    def run_no_network(self, *, image: str, name: str, env: Mapping[str, str]) -> str:
        if env:
            raise LiveAgentError("a live task container must not receive any env (no credential)")
        # Remove a stale container of the same name, then start detached with no
        # network and a long-lived shell so tools can exec into it.
        self._run(["rm", "-f", name], timeout=60)
        out = self._run(
            ["run", "-d", "--network", "none", "--name", name, "--entrypoint", "/bin/sh", image, "-c", "sleep infinity"],
            timeout=120,
        )
        if out.rc != 0:
            raise LiveAgentError(f"docker run failed: {out.stderr[-500:]}")
        return name

    def exec(self, *, container_id: str, argv: Sequence[str], stdin: bytes = b"") -> ExecOutcome:
        return self._run(["exec", "-i", container_id, *argv], stdin=stdin)

    def read_text(self, *, container_id: str, path: str) -> str | None:
        out = self._run(["exec", container_id, "/bin/sh", "-c", f"cat -- {path}"], timeout=60)
        return out.stdout if out.rc == 0 else None

    def stop(self, *, container_id: str) -> None:
        self._run(["rm", "-f", container_id], timeout=60)


__all__ = [
    "ContainerBackend",
    "DockerBackend",
    "ExecOutcome",
    "LiveAgentError",
    "build_arm_executor",
]
