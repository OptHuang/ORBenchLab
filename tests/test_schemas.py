"""The shipped JSON schemas, and the small validator that applies them."""

from __future__ import annotations

import json

import pytest

from orbenchlab.core import schema as schema_mod
from orbenchlab.core.schema import SchemaError, SchemaFeatureError

EXPECTED_SCHEMAS = {
    "normalized_rollout.schema.json",
    "inspection_report.schema.json",
    "plan_ledger.schema.json",
    "source_intake.schema.json",
}


def test_all_expected_schemas_ship():
    assert {p.name for p in schema_mod.iter_schema_paths()} == EXPECTED_SCHEMAS


def test_every_schema_loads_and_is_documented():
    for path in schema_mod.iter_schema_paths():
        schema = schema_mod.load_schema(path)
        assert schema.get("title"), path
        assert schema.get("description"), path
        assert schema.get("$id"), path


def test_a_schema_using_an_unimplemented_keyword_is_rejected(tmp_path):
    """A validator that silently skips a constraint reports success it did not verify."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"type": "object", "oneOf": []}), encoding="utf-8")
    with pytest.raises(SchemaFeatureError) as excinfo:
        schema_mod.load_schema(path)
    assert "oneOf" in str(excinfo.value)


def test_unimplemented_keywords_are_caught_in_nested_schemas(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps({"type": "object", "properties": {"a": {"type": "string", "allOf": []}}}),
        encoding="utf-8",
    )
    with pytest.raises(SchemaFeatureError):
        schema_mod.load_schema(path)


# --------------------------------------------------------------------------- #
# validator behaviour
# --------------------------------------------------------------------------- #


def test_missing_required_property_is_reported():
    schema = {"type": "object", "required": ["a"]}
    with pytest.raises(SchemaError) as excinfo:
        schema_mod.validate({}, schema)
    assert "missing required property 'a'" in str(excinfo.value)


def test_all_problems_are_reported_at_once():
    schema = {"type": "object", "required": ["a", "b", "c"]}
    message = str(pytest.raises(SchemaError, schema_mod.validate, {}, schema).value)
    assert message.count("missing required property") == 3


def test_type_mismatch_is_reported_with_a_pointer():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    with pytest.raises(SchemaError) as excinfo:
        schema_mod.validate({"n": "x"}, schema)
    assert "$.n: expected integer" in str(excinfo.value)


def test_booleans_are_not_accepted_as_integers():
    """bool subclasses int in Python; a schema saying integer must not take True."""
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    with pytest.raises(SchemaError):
        schema_mod.validate({"n": True}, schema)


def test_enum_const_pattern_and_bounds_are_enforced():
    schema = {
        "type": "object",
        "properties": {
            "e": {"enum": ["a", "b"]},
            "c": {"const": 1},
            "p": {"type": "string", "pattern": "^x+$"},
            "n": {"type": "integer", "minimum": 0, "maximum": 10},
        },
    }
    schema_mod.validate({"e": "a", "c": 1, "p": "xxx", "n": 5}, schema)
    for bad in ({"e": "z"}, {"c": 2}, {"p": "y"}, {"n": -1}, {"n": 11}):
        with pytest.raises(SchemaError):
            schema_mod.validate(bad, schema)


def test_nullable_types_are_supported():
    schema = {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
    schema_mod.validate({"x": None}, schema)
    schema_mod.validate({"x": "s"}, schema)
    with pytest.raises(SchemaError):
        schema_mod.validate({"x": 1}, schema)


def test_array_items_and_min_items_are_enforced():
    schema = {"type": "array", "minItems": 1, "items": {"type": "integer"}}
    schema_mod.validate([1, 2], schema)
    with pytest.raises(SchemaError):
        schema_mod.validate([], schema)
    with pytest.raises(SchemaError) as excinfo:
        schema_mod.validate([1, "x"], schema)
    assert "$[1]" in str(excinfo.value)


def test_additional_properties_false_is_enforced():
    schema = {"type": "object", "properties": {"a": {}}, "additionalProperties": False}
    schema_mod.validate({"a": 1}, schema)
    with pytest.raises(SchemaError) as excinfo:
        schema_mod.validate({"a": 1, "b": 2}, schema)
    assert "unexpected propert" in str(excinfo.value)


def test_the_inspection_schema_pins_the_static_read_guarantee():
    """The schema itself refuses a report claiming a model call or execution."""
    schema = schema_mod.load_schema(
        schema_mod.schemas_dir() / "inspection_report.schema.json"
    )
    execution = schema["properties"]["execution"]["properties"]
    assert execution["model_calls"]["maximum"] == 0
    assert execution["benchmark_executed"]["const"] is False
    assert execution["network_access"]["const"] is False
    assert execution["reads_credentials"]["const"] is False
