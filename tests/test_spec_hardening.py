"""Regression tests for campaign inputs that reach files or runner config."""

from __future__ import annotations

import pytest

from orbenchlab.campaign import spec as spec_mod
from orbenchlab.core.errors import SpecError


DIGEST = "sha256:" + "f" * 64


def _base(**overrides):
    raw = {
        "schema_version": "1.0",
        "slug": "hardening-probe",
        "date": "2026-08-24",
        "integration": "oragentbench",
        "site": "local-docker",
        "evidence_intent": "exploratory",
        "dataset": {"path": "ORAgentBench/harbor_tasks", "digest": DIGEST},
        "tasks": ["alpha"],
        "agents": [{"id": "oracle", "scaffold": "oracle"}],
        "budget": {"wall_clock_sec": 60, "max_cost_usd": 0},
        "seeds": [1],
        "attempts": 1,
        "shards": 1,
        "harbor": {"jobs_dir": "jobs"},
        "retry": {"max_retries": 0, "include_exceptions": []},
        "metrics": [],
    }
    raw.update(overrides)
    return raw


@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("budget.max_cost_usd", _base(budget={"wall_clock_sec": 60, "max_cost_usd": "free"})),
        ("budget.wall_clock_sec", _base(budget={"wall_clock_sec": "soon", "max_cost_usd": 0})),
        ("shards", _base(shards="many")),
        ("attempts", _base(attempts="one")),
        ("seeds[0]", _base(seeds=["first"])),
        ("retry.max_retries", _base(retry={"max_retries": "lots"})),
        ("harbor.n_attempts", _base(harbor={"jobs_dir": "jobs", "n_attempts": "one"})),
    ],
)
def test_malformed_numeric_fields_raise_a_diagnostic_spec_error(field, raw, sites_dir):
    with pytest.raises(SpecError) as excinfo:
        spec_mod.validate(raw, sites_dir=sites_dir)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("agent_id", ["../../../pwned", "a/b", "with space", ".", ".."])
def test_agent_id_cannot_become_a_path(agent_id, sites_dir):
    with pytest.raises(SpecError, match=r"agents\[0\]\.id"):
        spec_mod.validate(
            _base(agents=[{"id": agent_id, "scaffold": "oracle"}]), sites_dir=sites_dir
        )


def test_literal_environment_cannot_shadow_a_declared_secret(sites_dir):
    with pytest.raises(SpecError, match="overwrite"):
        spec_mod.validate(
            _base(
                agents=[
                    {
                        "id": "codex-safe",
                        "scaffold": "codex",
                        "model": "gpt-5.5",
                        "env_from_secret": {"OPENAI_API_KEY": "MODEL_API_KEY"},
                        "env_literals": {"OPENAI_API_KEY": "${OTHER_KEY}"},
                    }
                ],
                budget={"wall_clock_sec": 60, "max_cost_usd": 1},
            ),
            sites_dir=sites_dir,
        )


def test_provider_route_digest_must_be_content_addressed(sites_dir):
    with pytest.raises(SpecError, match="provider_route_digest"):
        spec_mod.validate(
            _base(
                agents=[
                    {
                        "id": "codex-safe",
                        "scaffold": "codex",
                        "model": "gpt-5.5",
                        "env_from_secret": {"OPENAI_API_KEY": "MODEL_API_KEY"},
                        "provider_route_digest": "not-a-digest",
                    }
                ],
                budget={"wall_clock_sec": 60, "max_cost_usd": 1},
            ),
            sites_dir=sites_dir,
        )
