"""The integration registry.

Adding a benchmark means adding a module with ``describe()`` and ``inspect()``
and registering it here. Nothing else in ORBenchLab needs to change: the CLI,
campaign validation and CI discover integrations through this registry.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from ..core.errors import IntegrationError
from . import frontieror, oragentbench
from .base import InspectionReport

#: Registration order is the display order of ``orbench integrations list``.
_MODULES: tuple[ModuleType, ...] = (oragentbench, frontieror)

_REGISTRY: dict[str, ModuleType] = {module.NAME: module for module in _MODULES}


def names() -> list[str]:
    return list(_REGISTRY)


def get(name: str) -> ModuleType:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise IntegrationError(
            f"unknown integration {name!r}; known integrations: {', '.join(names())}"
        ) from None


def describe(name: str) -> dict[str, Any]:
    return get(name).describe()


def describe_all() -> list[dict[str, Any]]:
    return [module.describe() for module in _MODULES]


def inspect(name: str, source: Path) -> InspectionReport:
    return get(name).inspect(Path(source))


def register(module: ModuleType) -> None:
    """Register an out-of-tree integration (used by tests and downstream forks)."""
    for attr in ("NAME", "describe", "inspect"):
        if not hasattr(module, attr):
            raise IntegrationError(f"integration module is missing required attribute {attr!r}")
    _REGISTRY[module.NAME] = module


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def summary_rows() -> list[dict[str, Any]]:
    """Compact rows for ``orbench integrations list``."""
    rows = []
    for description in describe_all():
        requires = description.get("requires", {})
        rows.append(
            {
                "name": description["name"],
                "kind": description["kind"],
                "upstream_repo": description["upstream_repo"],
                "pinned_commit": description["pinned_commit"],
                "performance_scored": description.get("performance_scored", False),
                "requires_model_api_key": requires.get("model_api_key", False),
                "requires_solver_license": requires.get("solver_license", False),
                "requires_self_hosted_runner": requires.get("self_hosted_runner", False),
                "secrets": requires.get("secrets", []),
                "runner_labels": requires.get("runner_labels", []),
            }
        )
    return rows


InspectFn = Callable[[Path], InspectionReport]
