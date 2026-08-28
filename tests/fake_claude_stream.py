"""A faithful fake of the Claude ``stream-json`` control protocol.

Run as a subprocess by the live-intervention state machine so tests exercise
the REAL async subprocess I/O and control handshake without a paid Claude call.
It reads stream-json user/control events on stdin and writes system/assistant/
control_response/result events on stdout, mimicking ``--replay-user-messages``.

Modes (argv[1]):
- ``honor``: emit a tool_use, honor the interrupt (ack + tool_result + interrupted
  result boundary), replay the hint, acknowledge the marker, final result.
- ``ignore-interrupt``: emit a tool_use and a result but never ack the interrupt
  (the naive path — the runner must refuse to count it).
- ``tool-in-flight``: ack the interrupt but reach the result boundary WITHOUT
  clearing the in-flight tool (the runner must block the hint on quiescence).
- ``baseline``: no tools; sid + assistant text + single result.
- ``leak-secret``: like ``honor`` but also prints the canary secret into the
  stream (the runner must scrub it from the journal).

argv[2]=session_id, argv[3]=hint_marker (echoed on replay), argv[4]=canary.
"""

from __future__ import annotations

import json
import sys


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def read() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def assistant_text(sid: str, text: str) -> dict:
    return {
        "type": "assistant",
        "session_id": sid,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def assistant_tool(sid: str, tool_id: str, name: str) -> dict:
    return {
        "type": "assistant",
        "session_id": sid,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {"cmd": "sleep"}}],
        },
    }


def tool_result(sid: str, tool_id: str, text: str) -> dict:
    return {
        "type": "user",
        "session_id": sid,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": text}],
        },
    }


def result(sid: str, subtype: str) -> dict:
    return {
        "type": "result",
        "subtype": subtype,
        "session_id": sid,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "total_cost_usd": 0.001,
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "honor"
    sid = sys.argv[2] if len(sys.argv) > 2 else "sid-fixed-1"
    marker = sys.argv[3] if len(sys.argv) > 3 else "MARKER"
    canary = sys.argv[4] if len(sys.argv) > 4 else ""

    # Consume the initial user turn.
    if read() is None:
        return 0
    emit({"type": "system", "subtype": "init", "session_id": sid})

    if mode == "baseline":
        emit(assistant_text(sid, "working on it"))
        emit(result(sid, "success"))
        return 0

    tool_id = "toolu_1"
    emit(assistant_tool(sid, tool_id, "Bash"))
    if canary:
        # Adversarial: try to leak the canary into the raw stream.
        emit(assistant_text(sid, f"debug secret={canary}"))

    # Wait for the interrupt control_request.
    control = read()
    if control is None:
        return 0
    request_id = (control.get("request_id") if isinstance(control, dict) else None) or "unknown"

    if mode == "ignore-interrupt":
        # Naive path: never ack; just finish the tool and end the turn.
        emit(tool_result(sid, tool_id, "SLOW_DONE"))
        emit(result(sid, "success"))
        return 0

    # Ack the interrupt.
    emit({"type": "control_response", "response": {"request_id": request_id, "subtype": "success"}})

    if mode == "tool-in-flight":
        # Reach the boundary WITHOUT clearing the pending tool_use.
        emit(result(sid, "interrupted"))
        return 0

    # honor / leak-secret: clear the tool, emit the interrupted boundary.
    emit(tool_result(sid, tool_id, "aborted-by-interrupt"))
    emit(result(sid, "interrupted"))

    # Read the hint turn, replay it (‑‑replay-user-messages), acknowledge, finish.
    hint = read()
    if hint is None:
        return 0
    hint_text = ""
    msg = hint.get("message") if isinstance(hint, dict) else None
    if isinstance(msg, dict):
        content = msg.get("content")
        hint_text = content if isinstance(content, str) else json.dumps(content)
    emit({"type": "user", "session_id": sid, "message": {"role": "user", "content": hint_text}})
    emit(assistant_text(sid, f"acknowledged {marker}; corrected"))
    emit(result(sid, "success"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
