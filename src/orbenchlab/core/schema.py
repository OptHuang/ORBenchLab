"""A deliberately small JSON Schema subset validator.

Pulling in a full validator would add a dependency for what amounts to shape
checking of files this repository also writes. What is supported is the subset
the shipped schemas use: ``type``, ``required``, ``properties``,
``additionalProperties`` (boolean form), ``items``, ``enum``, ``const``,
``minimum``, ``maximum``, ``minItems``, ``pattern``.

Anything a schema uses outside that subset raises :class:`SchemaFeatureError`
rather than being ignored — a validator that silently skips a constraint is
worse than no validator, because it reports success it did not verify.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .errors import ORBenchError

SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "description",
        "type",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "minItems",
        "pattern",
        "examples",
    }
)

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


class SchemaError(ORBenchError):
    """A document failed validation."""

    exit_code = 6


class SchemaFeatureError(ORBenchError):
    """A schema uses a keyword this validator does not implement."""

    exit_code = 7


def load_schema(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SchemaFeatureError(f"{path}: schema root must be an object")
    _assert_supported(data, path=str(path))
    return data


def _assert_supported(schema: Any, *, path: str, pointer: str = "#") -> None:
    if isinstance(schema, dict):
        unsupported = sorted(set(schema) - SUPPORTED_KEYWORDS)
        if unsupported:
            raise SchemaFeatureError(
                f"{path}{pointer}: unsupported schema keyword(s) {unsupported}; "
                f"supported subset: {sorted(SUPPORTED_KEYWORDS)}"
            )
        for key in ("properties",):
            for name, sub in (schema.get(key) or {}).items():
                _assert_supported(sub, path=path, pointer=f"{pointer}/{key}/{name}")
        if isinstance(schema.get("items"), dict):
            _assert_supported(schema["items"], path=path, pointer=f"{pointer}/items")


def validate(document: Any, schema: dict[str, Any], *, name: str = "document") -> None:
    """Validate ``document``; raise :class:`SchemaError` listing every problem."""
    problems: list[str] = []
    _validate(document, schema, "$", problems)
    if problems:
        bullets = "\n".join(f"  - {problem}" for problem in problems)
        raise SchemaError(f"{name} failed schema validation:\n{bullets}")


def _validate(value: Any, schema: dict[str, Any], pointer: str, problems: list[str]) -> None:
    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        allowed: tuple[type, ...] = tuple(t for name in types for t in _TYPE_MAP.get(name, ()))
        # bool is a subclass of int; an integer field must not accept True.
        if isinstance(value, bool) and "boolean" not in types:
            problems.append(f"{pointer}: expected {expected_type}, got boolean")
            return
        if allowed and not isinstance(value, allowed):
            problems.append(
                f"{pointer}: expected {expected_type}, got {type(value).__name__}"
            )
            return

    if "const" in schema and value != schema["const"]:
        problems.append(f"{pointer}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{pointer}: {value!r} not in enum {schema['enum']}")
    if "pattern" in schema and isinstance(value, str):
        if not re.search(schema["pattern"], value):
            problems.append(f"{pointer}: {value!r} does not match /{schema['pattern']}/")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        problems.append(f"{pointer}: {value} < minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        problems.append(f"{pointer}: {value} > maximum {schema['maximum']}")

    if isinstance(value, dict):
        _validate_object(value, schema, pointer, problems)
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{pointer}: {len(value)} items < minItems {schema['minItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{pointer}[{index}]", problems)


def _validate_object(
    value: dict[str, Any], schema: dict[str, Any], pointer: str, problems: list[str]
) -> None:
    properties: dict[str, Any] = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in value:
            problems.append(f"{pointer}: missing required property {key!r}")
    if schema.get("additionalProperties") is False:
        extra = sorted(set(value) - set(properties))
        if extra:
            problems.append(f"{pointer}: unexpected propert(ies) {extra}")
    for key, sub_schema in properties.items():
        if key in value and isinstance(sub_schema, dict):
            _validate(value[key], sub_schema, f"{pointer}.{key}", problems)


def schemas_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "schemas"


def iter_schema_paths() -> Iterable[Path]:
    return sorted(schemas_dir().glob("*.json"))
