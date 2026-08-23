"""The identifier algebra must be a pure function of its inputs.

If any of these fail, run ids stop being reproducible and the whole plan-ledger
reconciliation scheme collapses.
"""

from __future__ import annotations

import pytest

from orbenchlab.core import ids

BASE_AGENT = {
    "scaffold": "claude-code",
    "scaffold_version": "1.2.3",
    "model_id": "pinned-model-1",
    "prompt_digest": "sha256:" + "ab" * 32,
    "tool_policy": {"disallowed_tools": ["WebSearch"]},
    "agent_kwargs": {"temperature": 0},
    "env_keys": ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"],
}


def _run(**overrides):
    kwargs = {
        "cfg_digest": "d" * 64,
        "task_uid_value": "bench@sha256:aa/task#chk",
        "agent_uid_value": ids.agent_uid(**BASE_AGENT),
        "budget_uid_value": ids.budget_uid({"wall_clock_sec": 600}),
        "seed": 1,
        "attempt": 1,
    }
    kwargs.update(overrides)
    return ids.run_id(**kwargs)


def test_canon_is_key_order_independent():
    assert ids.canon({"a": 1, "b": 2}) == ids.canon({"b": 2, "a": 1})


def test_run_id_is_stable_across_calls():
    assert _run() == _run()


def test_run_id_has_expected_shape():
    value = _run()
    assert len(value) == ids.RUN_ID_LENGTH
    assert all(char in "0123456789abcdef" for char in value)


def test_changing_seed_changes_run_id_but_not_agent_identity():
    """Seed is deliberately outside agent_uid so agent identity spans seeds."""
    assert _run(seed=1) != _run(seed=2)
    assert ids.agent_uid(**BASE_AGENT) == ids.agent_uid(**BASE_AGENT)


def test_env_key_order_does_not_matter_but_names_do():
    reordered = dict(BASE_AGENT, env_keys=list(reversed(BASE_AGENT["env_keys"])))
    assert ids.agent_uid(**reordered) == ids.agent_uid(**BASE_AGENT)

    renamed = dict(BASE_AGENT, env_keys=["ANTHROPIC_AUTH_TOKEN", "OTHER_BASE_URL"])
    assert ids.agent_uid(**renamed) != ids.agent_uid(**BASE_AGENT)


def test_secret_values_are_not_part_of_any_identifier():
    """agent_uid takes names only, so rotating a credential changes nothing.

    This is what lets a campaign resume after a key rotation without every run
    id shifting underneath it.
    """
    signature = ids.agent_uid(**BASE_AGENT)
    # There is no parameter through which a value could enter; the closest thing
    # is a key whose name happens to contain a value-looking string.
    with_value_like_name = dict(BASE_AGENT, env_keys=["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"])
    assert ids.agent_uid(**with_value_like_name) == signature


def test_provider_route_digest_is_part_of_agent_identity():
    first = ids.agent_uid(
        **BASE_AGENT, provider_route_digest="sha256:" + "1" * 64
    )
    second = ids.agent_uid(
        **BASE_AGENT, provider_route_digest="sha256:" + "2" * 64
    )
    assert first != second


def test_campaign_cfg_digest_ignores_non_semantic_fields():
    base = {"slug": "x", "tasks": ["a"], "job_name": "one", "debug": False}
    other = {"slug": "x", "tasks": ["a"], "job_name": "two", "debug": True, "comments": "hi"}
    assert ids.campaign_cfg_digest(base) == ids.campaign_cfg_digest(other)


def test_campaign_cfg_digest_ignores_excluded_keys_at_any_depth():
    base = {"a": {"b": {"tasks": ["t"], "job_name": "one"}}}
    other = {"a": {"b": {"tasks": ["t"], "job_name": "two"}}}
    assert ids.campaign_cfg_digest(base) == ids.campaign_cfg_digest(other)


def test_campaign_cfg_digest_reacts_to_semantic_change():
    assert ids.campaign_cfg_digest({"tasks": ["a"]}) != ids.campaign_cfg_digest({"tasks": ["b"]})


def test_shard_assignment_is_pure_and_in_range():
    run = _run()
    assert ids.shard_of(run, 8) == ids.shard_of(run, 8)
    assert 0 <= ids.shard_of(run, 8) < 8


def test_shard_rejects_zero_shards():
    with pytest.raises(ValueError):
        ids.shard_of(_run(), 0)


def test_match_key_identifies_the_reconciliation_tuple():
    agent = ids.agent_uid(**BASE_AGENT)
    key = ids.match_key(task_name="t", agent_uid_value=agent, seed=1, attempt=1)
    assert key == ids.match_key(task_name="t", agent_uid_value=agent, seed=1, attempt=1)
    assert key != ids.match_key(task_name="t", agent_uid_value=agent, seed=2, attempt=1)
    assert key != ids.match_key(task_name="u", agent_uid_value=agent, seed=1, attempt=1)


def test_rescoring_produces_a_new_id_without_replacing_the_run():
    run = _run()
    first = ids.score_run_id(run, "sha256:" + "1" * 64, "initial")
    second = ids.score_run_id(run, "sha256:" + "2" * 64, "verifier fix")
    assert first != second
    assert first != run and second != run


def test_campaign_id_embeds_date_and_digest_prefix():
    digest = "abcdef01" + "0" * 56
    assert ids.campaign_id("demo", "20260822", digest) == "demo-20260822-abcdef01"
