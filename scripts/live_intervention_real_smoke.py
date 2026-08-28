#!/usr/bin/env python3
"""Minimal REAL Volc live-intervention smoke (paid, tiny budget).

Runs one baseline and one L1 arm of the live-intervention study against a
trivial verifier-grounded task, using the real Docker no-network container, the
real P0-A credential relay to the real Volc API, and the real Claude CLI on the
host with tools confined to the container MCP proxy.

Success criteria (printed as JSON, canary never echoed):
- baseline arm reaches a result; L1 arm shows interrupt -> ack -> interrupted
  boundary -> hint replayed (protocol_satisfied);
- each arm is graded by the container verifier (reward present);
- the L1 hint marker appears in the raw stream and ATIF;
- the provider-secret canary is absent from every arm journal/ATIF/artifact and
  from the task container env/proc.

Env: reads ARK creds from ~/.config/claude/ark-volces.env
     (ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL).
Usage: python3 scripts/live_intervention_real_smoke.py [out_dir]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbenchlab import harbor_container_proxy  # noqa: E402
from orbenchlab import harbor_credentials  # noqa: E402
from orbenchlab import harbor_live_agent as hla  # noqa: E402
from orbenchlab import harbor_live_intervention as hli  # noqa: E402

# The MCP proxy is launched by Claude with Claude's minimal env, so run it by
# absolute file path (it is stdlib-only and self-contained), not `-m package`.
PROXY_PATH = os.path.abspath(harbor_container_proxy.__file__)

CLAUDE = os.path.expanduser(
    "~/.local/node/node-v22.12.0-linux-x64/bin/claude"
)


def _load_creds() -> dict[str, str]:
    path = Path(os.path.expanduser("~/.config/claude/ark-volces.env"))
    creds: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        creds[key] = value.strip().strip('"').strip("'")
    return creds


def _write_task(root: Path) -> Path:
    """A trivial verifier-grounded task: write the required word to answer.txt."""

    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN mkdir -p /work /logs/verifier\nCMD [\"/bin/sh\"]\n"
    )
    (task / "instruction.md").write_text(
        "Use the bash tool to write the single word BASELINE into /work/answer.txt "
        "(run: echo BASELINE > /work/answer.txt), then stop. Do not write anything else."
    )
    return task


# Self-contained verifier: no /tests mount needed in the container.
VERIFIER_ARGV = [
    "/bin/sh",
    "-c",
    "mkdir -p /logs/verifier; "
    "got=$(cat /work/answer.txt 2>/dev/null | tr -d '\\n \\t'); "
    "if [ \"$got\" = HINTED ]; then printf '1.0\\n' > /logs/verifier/reward.txt; "
    "else printf '0.0\\n' > /logs/verifier/reward.txt; fi; "
    "printf '{\"summary\":{\"got\":\"%s\"}}' \"$got\" > /logs/verifier/ctrf.json",
]


def main() -> int:
    creds = _load_creds()
    base_url = creds.get("ANTHROPIC_BASE_URL", "")
    token = creds.get("ANTHROPIC_AUTH_TOKEN", "")
    model = creds.get("ANTHROPIC_MODEL") or creds.get("ANTHROPIC_MODEL_ID") or "doubao-seed-1-6-250615"
    if not base_url or not token:
        print(json.dumps({"ok": False, "error": "missing ARK creds"}))
        return 1

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="orbench-live-real-"))
    out.mkdir(parents=True, exist_ok=True)
    task = _write_task(out)

    @contextmanager
    def relay_factory():
        relay = harbor_credentials.CredentialRelay(real_base_url=base_url, real_token=token)
        with relay:
            yield relay

    def make_policy(level: str) -> hli.InterventionPolicy:
        if level == "baseline":
            return hli.InterventionPolicy(level="baseline")
        marker = "LIVE_HINT_MARKER_REAL"
        return hli.InterventionPolicy(
            level=level,
            hint_text=(
                f"{marker}: correction — the required word is HINTED, not BASELINE. "
                "Run: echo HINTED > /work/answer.txt using the bash tool, then stop."
            ),
            hint_marker=marker,
            trigger=hli.Trigger("tool-use"),
        )

    executor = hla.build_arm_executor(
        task_dir=task,
        backend=hla.DockerBackend(),
        relay_factory=relay_factory,
        claude_executable=CLAUDE,
        proxy_argv_prefix=[sys.executable, PROXY_PATH],
        verifier_argv=VERIFIER_ARGV,
        model=model,
        max_turns=6,
        max_budget_usd=0.15,
        secret_values=[token],
        make_policy=make_policy,
    )

    # Minimal spend: one baseline arm and one L1 arm run directly (not the full
    # >=5-repeat study), enough to prove the end-to-end path and the hint effect.
    arms = {}
    for level in ("baseline", "L1"):
        arm = executor(level, 1, f"smoke-{level}", out / "arms" / level)
        journal = arm.get("journal", {})
        arms[level] = {
            "protocol_satisfied": journal.get("protocol_satisfied"),
            "interrupt": journal.get("interrupt"),
            "hint": journal.get("hint"),
            "single_session": journal.get("single_session"),
            "claude_session_id": journal.get("claude_session_id"),
            "reward": arm.get("reward"),
            "ctrf": arm.get("ctrf"),
            "error": journal.get("error"),
        }
    leaked = harbor_credentials.scan_tree_for_secret(out, [token])
    l1 = arms["L1"]
    summary = {
        "ok": bool(
            l1["protocol_satisfied"]
            and l1["interrupt"] and l1["interrupt"].get("acked")
            and l1["hint"] and l1["hint"].get("replayed")
            and arms["baseline"].get("reward") is not None
            and l1.get("reward") is not None
            and not leaked
        ),
        "baseline": arms["baseline"],
        "L1": l1,
        "hint_recovered": (arms["baseline"].get("reward"), l1.get("reward")),
        "secret_leaked_paths": leaked,
        "out": str(out),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
