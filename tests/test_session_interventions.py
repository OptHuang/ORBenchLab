from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbenchlab import session_interventions

VOLC = "https://ark.cn-beijing.volces.com/api/coding"
PROVIDER = {"ANTHROPIC_BASE_URL": VOLC, "ANTHROPIC_AUTH_TOKEN": "fixture-secret"}
POLICY = {
    "trigger": {"kind": "assistant-event-index", "value": 2},
    "hint_level": 1,
    "hint_text": "Hint: the answer is in data/instance.json.",
}


def _fake_claude(path: Path) -> Path:
    """Contract-faithful CLI: echoes injected user turns and continues."""

    executable = path / "fake-claude"
    executable.write_text(
        """#!/usr/bin/env python3
import json, sys

SID = "fixture-session-1234"
line = sys.stdin.readline()
if not line:
    raise SystemExit(1)
print(json.dumps({"type": "system", "subtype": "init", "session_id": SID}), flush=True)
print(json.dumps({"type": "assistant", "session_id": SID, "message": {"content": "thinking step 1"}}), flush=True)
print(json.dumps({"type": "assistant", "session_id": SID, "message": {"content": "thinking step 2"}}), flush=True)
hint = sys.stdin.readline()
if hint:
    payload = json.loads(hint)
    content = payload["message"]["content"]
    with open("hint.txt", "w", encoding="utf-8") as stream:
        stream.write(content)
    print(json.dumps({"type": "user", "session_id": SID, "message": {"role": "user", "content": content}}), flush=True)
    print(json.dumps({"type": "assistant", "session_id": SID, "message": {"content": "using the hint"}}), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "session_id": SID, "total_cost_usd": 0.01}), flush=True)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _adversarial_claude(path: Path, *, mode: str) -> Path:
    """Reproduces the independent reviewer's false-confirmation exploit.

    ``ignore-hint``: emits two stale assistant events (firing the policy),
    then one user event that does NOT contain the hint, never reads stdin,
    and exits 7.  ``echo-no-assistant``: consumes and echoes the hint but
    never produces an assistant continuation.
    """

    executable = path / f"adversarial-claude-{mode}"
    executable.write_text(
        f"""#!/usr/bin/env python3
import json, sys, time

mode = {mode!r}
SID = "fixture-session-1234"
line = sys.stdin.readline()
if not line:
    raise SystemExit(1)
print(json.dumps({{"type": "system", "subtype": "init", "session_id": SID}}), flush=True)
time.sleep(0.2)
print(json.dumps({{"type": "assistant", "session_id": SID, "message": {{"content": "stale step 1"}}}}), flush=True)
print(json.dumps({{"type": "assistant", "session_id": SID, "message": {{"content": "stale step 2"}}}}), flush=True)
if mode == "ignore-hint":
    time.sleep(0.5)
    print(json.dumps({{"type": "user", "session_id": SID, "message": {{"role": "user", "content": "unrelated tool result"}}}}), flush=True)
    raise SystemExit(7)
hint = sys.stdin.readline()
if hint:
    payload = json.loads(hint)
    print(json.dumps({{"type": "user", "session_id": SID, "message": {{"role": "user", "content": payload["message"]["content"]}}}}), flush=True)
print(json.dumps({{"type": "result", "subtype": "success", "session_id": SID, "total_cost_usd": 0.01}}), flush=True)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _verifier(path: Path) -> Path:
    script = path / "verify.sh"
    script.write_text("#!/bin/sh\ntest -f hint.txt\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def test_capability_probe_is_fail_closed_and_machine_readable():
    claude = session_interventions.probe_capability(profile="claude-code")
    assert claude["same_session_hint_injection"] is True
    assert claude["injection_contract"] == session_interventions.INJECTION_CONTRACT
    codex = session_interventions.probe_capability(profile="codex")
    assert codex["same_session_hint_injection"] is False
    harbor = session_interventions.probe_capability(
        profile="claude-code", runtime="harbor-trial"
    )
    assert harbor["same_session_hint_injection"] is False
    assert "restart-with-hint" in harbor["reason"]
    for receipt in (claude, codex, harbor):
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_digest"}
        assert receipt["receipt_digest"] == session_interventions._digest(unsigned)


def test_codex_injection_is_refused_not_downgraded(tmp_path: Path):
    with pytest.raises(session_interventions.SessionInterventionError):
        session_interventions.run_intervention_session(
            profile="codex",
            stage="s",
            model="m",
            prompt="p",
            workdir=tmp_path,
            out=tmp_path / "out",
            timeout_sec=5,
            environ={"OPENAI_BASE_URL": VOLC, "OPENAI_API_KEY": "k"},
            max_budget_usd=0.1,
            policy=POLICY,
        )


def test_same_session_injection_fires_and_is_confirmed(tmp_path: Path):
    executable = _fake_claude(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    receipt = session_interventions.run_intervention_session(
        profile="claude-code",
        stage="contract-test/treatment",
        model="fixture-model",
        prompt="Solve the task.",
        workdir=workdir,
        out=tmp_path / "sessions",
        timeout_sec=20,
        environ=PROVIDER,
        max_budget_usd=0.1,
        policy=POLICY,
        executable=executable,
    )
    assert receipt["status"] == "completed"
    injection = receipt["injection"]
    assert injection["fired"] is True
    assert injection["injection_confirmed"] is True
    assert injection["pre_injection_assistant_events"] == 2
    assert injection["post_injection_events"] >= 1
    assert injection["fired_at_sec"] is not None
    assert receipt["intervention_class"] == "same-session-continuation"
    assert (workdir / "hint.txt").read_text() == POLICY["hint_text"]
    events = [
        json.loads(line)
        for line in (
            Path(receipt["receipt_path"]).parent / "events.jsonl"
        ).read_text().splitlines()
    ]
    assert [event["type"] for event in events][:3] == ["system", "assistant", "assistant"]
    assert all("at_sec" in event for event in events)


def test_control_session_closes_stdin_without_injection(tmp_path: Path):
    executable = _fake_claude(tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir()
    receipt = session_interventions.run_intervention_session(
        profile="claude-code",
        stage="contract-test/control",
        model="fixture-model",
        prompt="Solve the task.",
        workdir=workdir,
        out=tmp_path / "sessions",
        timeout_sec=20,
        environ=PROVIDER,
        max_budget_usd=0.1,
        policy=None,
        executable=executable,
    )
    assert receipt["status"] == "completed"
    assert receipt["injection"]["requested"] is False
    assert receipt["injection"]["fired"] is False
    assert receipt["intervention_class"] == "none"
    assert not (workdir / "hint.txt").exists()


def test_controlled_study_reaches_e4_only_with_confirmed_paired_arms(tmp_path: Path):
    executable = _fake_claude(tmp_path)
    template = tmp_path / "template"
    template.mkdir()
    (template / "data.json").write_text("{}", encoding="utf-8")
    verifier = _verifier(tmp_path)
    study = session_interventions.run_intervention_study(
        profile="claude-code",
        model="fixture-model",
        prompt="Solve the task.",
        template_workdir=template,
        out=tmp_path / "study",
        environ=PROVIDER,
        verifier_argv=[str(verifier)],
        policy=POLICY,
        n_control=3,
        n_treatment=3,
        timeout_sec=20,
        max_budget_usd=0.1,
        executable=executable,
    )
    assert study["status"] == "completed"
    assert study["evidence_level"] == "E4-controlled-same-session-intervention"
    assert study["all_treatment_injections_confirmed"] is True
    assert study["arms"]["treatment"]["pass_rate"] == 1.0
    assert study["arms"]["control"]["pass_rate"] == 0.0
    assert len(study["trials"]) == 6
    written = json.loads(
        (tmp_path / "study" / "intervention-study.json").read_text(encoding="utf-8")
    )
    assert written["receipt_digest"] == study["receipt_digest"]


def test_underpowered_study_never_claims_e4(tmp_path: Path):
    executable = _fake_claude(tmp_path)
    template = tmp_path / "template"
    template.mkdir()
    verifier = _verifier(tmp_path)
    study = session_interventions.run_intervention_study(
        profile="claude-code",
        model="fixture-model",
        prompt="Solve the task.",
        template_workdir=template,
        out=tmp_path / "study",
        environ=PROVIDER,
        verifier_argv=[str(verifier)],
        policy=POLICY,
        n_control=1,
        n_treatment=1,
        timeout_sec=20,
        max_budget_usd=0.1,
        executable=executable,
    )
    assert study["evidence_level"] == "E3-underpowered-same-session-intervention"


def test_adversarial_ignore_hint_study_never_reaches_e4(tmp_path: Path):
    # Reproduces the independent reviewer's exploit: stale assistant events fire
    # the policy, a non-hint user event appears, the hint is never consumed and
    # the session exits 7. All six sessions fail; must be E1, no continuation.
    executable = _adversarial_claude(tmp_path, mode="ignore-hint")
    template = tmp_path / "template"
    template.mkdir()
    (template / "seed.json").write_text("{}", encoding="utf-8")
    verifier = tmp_path / "always-fail.sh"
    verifier.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    verifier.chmod(0o755)
    study = session_interventions.run_intervention_study(
        profile="claude-code",
        model="fixture-model",
        prompt="Solve the task.",
        template_workdir=template,
        out=tmp_path / "study",
        environ=PROVIDER,
        verifier_argv=[str(verifier)],
        policy=POLICY,
        n_control=3,
        n_treatment=3,
        timeout_sec=20,
        max_budget_usd=0.1,
        executable=executable,
    )
    assert study["evidence_level"] == "E1-incomplete-intervention-evidence"
    assert study["all_treatment_injections_confirmed"] is False
    treatment = [row for row in study["trials"] if row["arm"] == "treatment"]
    assert treatment and all(
        row["intervention_class"] != "same-session-continuation" for row in treatment
    )


def test_adversarial_echo_without_assistant_never_confirms(tmp_path: Path):
    executable = _adversarial_claude(tmp_path, mode="echo-no-assistant")
    workdir = tmp_path / "work"
    workdir.mkdir()
    receipt = session_interventions.run_intervention_session(
        profile="claude-code",
        stage="adv/echo",
        model="fixture-model",
        prompt="Solve.",
        workdir=workdir,
        out=tmp_path / "sessions",
        timeout_sec=20,
        environ=PROVIDER,
        max_budget_usd=0.1,
        policy=POLICY,
        executable=executable,
    )
    # The hint is echoed but no assistant continuation follows it.
    assert receipt["injection"]["hint_echo_event_index"] is not None
    assert receipt["injection"]["first_post_hint_assistant_index"] is None
    assert receipt["injection"]["injection_confirmed"] is False
    assert receipt["intervention_class"] == "none"


def test_tampered_session_receipt_blocks_e4(tmp_path: Path):
    executable = _fake_claude(tmp_path)
    template = tmp_path / "template"
    template.mkdir()
    verifier = _verifier(tmp_path)
    out = tmp_path / "study"
    study = session_interventions.run_intervention_study(
        profile="claude-code",
        model="fixture-model",
        prompt="Solve the task.",
        template_workdir=template,
        out=out,
        environ=PROVIDER,
        verifier_argv=[str(verifier)],
        policy=POLICY,
        n_control=3,
        n_treatment=3,
        timeout_sec=20,
        max_budget_usd=0.1,
        executable=executable,
    )
    assert study["evidence_level"] == "E4-controlled-same-session-intervention"
    # Tamper with one sealed receipt (flip a control to a fake continuation)
    # and rebuild the study from the same root.
    receipt_files = sorted(out.glob("trials/control-01/sessions/*/receipt.json"))
    assert receipt_files
    doc = json.loads(receipt_files[0].read_text())
    assert doc["intervention_class"] == "none"
    doc["intervention_class"] = "same-session-continuation"
    receipt_files[0].write_text(json.dumps(doc), encoding="utf-8")
    # Rebuilding over a tampered receipt must fail closed, not silently
    # re-run and overwrite it back to a passing E4.
    with pytest.raises(session_interventions.SessionInterventionError):
        session_interventions.run_intervention_study(
            profile="claude-code",
            model="fixture-model",
            prompt="Solve the task.",
            template_workdir=template,
            out=out,
            environ=PROVIDER,
            verifier_argv=[str(verifier)],
            policy=POLICY,
            n_control=3,
            n_treatment=3,
            timeout_sec=20,
            max_budget_usd=0.1,
            executable=executable,
        )


def test_study_resume_does_not_rebuy_sessions(tmp_path: Path):
    # The idempotency invariant behind P0-3: a second run over the same root
    # must launch zero new agent processes (marker count stays constant).
    marker = tmp_path / "invocations.log"
    executable = tmp_path / "counting-claude"
    executable.write_text(
        f"""#!/usr/bin/env python3
import json, sys
with open({str(marker)!r}, "a") as fh:
    fh.write("call\\n")
SID = "fixture-session-1234"
line = sys.stdin.readline()
print(json.dumps({{"type": "system", "subtype": "init", "session_id": SID}}), flush=True)
print(json.dumps({{"type": "assistant", "session_id": SID, "message": {{"content": "s1"}}}}), flush=True)
print(json.dumps({{"type": "assistant", "session_id": SID, "message": {{"content": "s2"}}}}), flush=True)
hint = sys.stdin.readline()
if hint:
    content = json.loads(hint)["message"]["content"]
    open("hint.txt", "w").write(content)
    print(json.dumps({{"type": "user", "session_id": SID, "message": {{"role": "user", "content": content}}}}), flush=True)
    print(json.dumps({{"type": "assistant", "session_id": SID, "message": {{"content": "using hint"}}}}), flush=True)
print(json.dumps({{"type": "result", "subtype": "success", "session_id": SID, "total_cost_usd": 0.01}}), flush=True)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    template = tmp_path / "template"
    template.mkdir()
    verifier = _verifier(tmp_path)
    kwargs = dict(
        profile="claude-code",
        model="fixture-model",
        prompt="Solve the task.",
        template_workdir=template,
        out=tmp_path / "study",
        environ=PROVIDER,
        verifier_argv=[str(verifier)],
        policy=POLICY,
        n_control=3,
        n_treatment=3,
        timeout_sec=20,
        max_budget_usd=0.1,
        executable=executable,
    )
    session_interventions.run_intervention_study(**kwargs)
    first = marker.read_text().count("call")
    assert first == 6
    session_interventions.run_intervention_study(**kwargs)
    assert marker.read_text().count("call") == 6


def test_unsupported_profile_study_is_recorded_not_faked(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    study = session_interventions.run_intervention_study(
        profile="codex",
        model="fixture-model",
        prompt="Solve.",
        template_workdir=template,
        out=tmp_path / "study",
        environ={"OPENAI_BASE_URL": VOLC, "OPENAI_API_KEY": "k"},
        verifier_argv=["true"],
        policy=POLICY,
    )
    assert study["status"] == "unsupported-capability"
    assert study["evidence_level"] == "E0-unsupported"
    assert study["trials"] == []
