"""Exception types. Everything that cannot be verified fails closed."""

from __future__ import annotations


class ORBenchError(Exception):
    """Base class for all ORBenchLab errors."""

    exit_code = 1


class SpecError(ORBenchError):
    """A campaign spec is malformed or violates a policy check."""

    exit_code = 2


class IntegrationError(ORBenchError):
    """An integration could not be resolved or inspected."""

    exit_code = 3


class EvidenceError(ORBenchError):
    """A report asked for a claim its evidence cannot support."""

    exit_code = 4


class PreconditionError(ORBenchError):
    """A required secret, license, runner or upstream source is absent.

    Raised instead of degrading silently: an absent Gurobi license or model key
    must stop the run, not produce a report that looks like a real measurement.
    """

    exit_code = 5
