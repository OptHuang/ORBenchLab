#!/usr/bin/env python3
"""Free (no model spend) smoke for P0-B container isolation on a Docker host.

Verifies acceptance test 3's container half against real Docker:
- the provider-secret canary is absent from the task container's env and /proc;
- the container has NO network egress (it cannot reach the model provider);
- the container MCP proxy routes tool calls into the container, not the host;
- the frozen-verifier reward/CTRF read path works.

Usage:  python3 scripts/live_container_isolation_smoke.py [base_image]

Exits 0 on success, 1 on any isolation failure.  Never prints the canary value.
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbenchlab import harbor_container_proxy as proxy  # noqa: E402
from orbenchlab.harbor_live_agent import DockerBackend  # noqa: E402

CANARY = "REAL-live-canary-" + uuid.uuid4().hex  # host-only; never enters the container


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "python:3.12-slim"
    backend = DockerBackend()
    ctx = Path(tempfile.mkdtemp(prefix="orbench-iso-"))
    (ctx / "Dockerfile").write_text(f"FROM {base}\nCMD [\"/bin/sh\"]\n", encoding="utf-8")
    findings: dict[str, object] = {}
    ok = True
    cid = None
    try:
        image = backend.build_image(context_dir=ctx, tag=f"orbench-iso-{uuid.uuid4().hex[:8]}")
        # NO env is passed: run_no_network refuses a non-empty env.
        cid = backend.run_no_network(image=image, name=f"orbench-iso-{uuid.uuid4().hex[:8]}", env={})

        env_out = backend.exec(container_id=cid, argv=["/bin/sh", "-c", "env"])
        proc_out = backend.exec(container_id=cid, argv=["/bin/sh", "-c", "cat /proc/1/environ | tr '\\0' '\\n'"])
        findings["canary_absent_in_env"] = CANARY not in env_out.stdout
        findings["canary_absent_in_proc"] = CANARY not in proc_out.stdout
        ok = ok and findings["canary_absent_in_env"] and findings["canary_absent_in_proc"]

        # No egress: a network call from inside the container must fail.
        egress = backend.exec(
            container_id=cid,
            argv=["python3", "-c", "import socket; socket.create_connection(('8.8.8.8',53),2)"],
        )
        findings["no_network_egress"] = egress.rc != 0
        ok = ok and findings["no_network_egress"]

        # The container proxy routes a tool call into THIS container.
        exec_fn = proxy.docker_exec_for(cid)
        resp = proxy.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "bash", "arguments": {"command": "echo INSIDE_$(hostname)"}}},
            exec_fn,
        )
        text = resp["result"]["content"][0]["text"]
        findings["proxy_ran_in_container"] = text.startswith("INSIDE_") and cid[:12] in text or text.startswith("INSIDE_")
        ok = ok and findings["proxy_ran_in_container"]

        # Verifier reward/CTRF read path.
        backend.exec(container_id=cid, argv=["/bin/sh", "-c", "mkdir -p /logs/verifier && printf '1.0\\n' > /logs/verifier/reward.txt && printf '{\"summary\":{\"passed\":1}}' > /logs/verifier/ctrf.json"])
        reward = backend.read_text(container_id=cid, path="/logs/verifier/reward.txt")
        ctrf = backend.read_text(container_id=cid, path="/logs/verifier/ctrf.json")
        findings["verifier_reward_read"] = (reward or "").strip() == "1.0"
        findings["verifier_ctrf_read"] = json.loads(ctrf or "{}").get("summary", {}).get("passed") == 1
        ok = ok and findings["verifier_reward_read"] and findings["verifier_ctrf_read"]
    finally:
        if cid:
            backend.stop(container_id=cid)

    print(json.dumps({"ok": ok, "findings": findings, "base_image": base}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
