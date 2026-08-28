"""A minimal MCP stdio proxy that confines an agent's tools to the Harbor task
container.

The live-intervention agent runs with every built-in tool (Bash/Read/Write/
Edit/Web) disabled and only this proxy's tools allowed.  Each tool call is
executed *inside* the Harbor ``BaseEnvironment`` task container through an
injected ``ContainerExec`` callable — never on the host and never with network
egress.  The proxy holds no provider credential and opens no host file or
socket itself, so a compromised agent cannot read the host or reach the model
provider through its tools.

The JSON-RPC handling core (``handle_request``) is pure and unit-tested with a
fake exec function; ``serve`` runs the stdio loop, and ``__main__`` builds the
real ``docker exec`` function for a given container id.
"""

from __future__ import annotations

import json
import shlex
import sys
from typing import Any, Callable, Mapping

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "orbench"

# exec_fn(argv, stdin_bytes) -> {"rc": int, "stdout": str, "stderr": str}
ContainerExec = Callable[[list[str], bytes], Mapping[str, Any]]


class ContainerProxyError(Exception):
    pass


_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command INSIDE the no-network task container.",
        "inputSchema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from inside the task container.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a file inside the task container.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
]


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _call_tool(name: str, arguments: Mapping[str, Any], exec_fn: ContainerExec) -> dict[str, Any]:
    if name == "bash":
        command = str(arguments.get("command", ""))
        if not command:
            return _text_result("bash requires a command", is_error=True)
        # Always route through the container; the proxy never runs a host shell.
        out = exec_fn(["/bin/sh", "-c", command], b"")
    elif name == "read_file":
        path = str(arguments.get("path", ""))
        if not path:
            return _text_result("read_file requires a path", is_error=True)
        out = exec_fn(["/bin/sh", "-c", f"cat -- {shlex.quote(path)}"], b"")
    elif name == "write_file":
        path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        if not path:
            return _text_result("write_file requires a path", is_error=True)
        out = exec_fn(["/bin/sh", "-c", f"cat > {shlex.quote(path)}"], content.encode())
    else:
        return _text_result(f"unknown tool {name!r}", is_error=True)
    rc = int(out.get("rc", 1))
    stdout = str(out.get("stdout", ""))
    stderr = str(out.get("stderr", ""))
    body = stdout if rc == 0 else f"exit={rc}\n{stdout}\n{stderr}"
    return _text_result(body, is_error=rc != 0)


def handle_request(request: Mapping[str, Any], exec_fn: ContainerExec) -> dict[str, Any] | None:
    """Handle one JSON-RPC request; return the response (or None for notifications)."""

    method = request.get("method")
    req_id = request.get("id")
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": _TOOLS}
    elif method == "tools/call":
        params = request.get("params") or {}
        result = _call_tool(str(params.get("name")), params.get("arguments") or {}, exec_fn)
    elif method in {"notifications/initialized", "initialized"} or req_id is None:
        return None
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "method not found"}}
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def serve(exec_fn: ContainerExec, reader=None, writer=None) -> int:
    reader = reader if reader is not None else sys.stdin
    writer = writer if writer is not None else sys.stdout
    for line in reader:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_request(request, exec_fn)
        if response is not None:
            writer.write(json.dumps(response, ensure_ascii=False) + "\n")
            writer.flush()
    return 0


def docker_exec_for(container_id: str, docker_bin: str = "docker") -> ContainerExec:
    """Build a ContainerExec that runs commands inside a running container.

    The container is expected to be started with no network (``--network none``)
    by Harbor's BaseEnvironment, so tool calls cannot reach the model provider.
    """

    import subprocess

    def exec_fn(argv: list[str], stdin_bytes: bytes) -> dict[str, Any]:
        proc = subprocess.run(
            [docker_bin, "exec", "-i", container_id, *argv],
            input=stdin_bytes,
            capture_output=True,
            timeout=600,
        )
        return {
            "rc": proc.returncode,
            "stdout": proc.stdout.decode("utf-8", errors="replace"),
            "stderr": proc.stderr.decode("utf-8", errors="replace"),
        }

    return exec_fn


def _main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: harbor_container_proxy <container_id>\n")
        return 2
    return serve(docker_exec_for(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(_main())
