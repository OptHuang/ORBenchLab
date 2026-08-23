"""ORBenchLab — a control plane for operations-research agent benchmarks.

ORBenchLab registers pinned upstream integrations, compiles campaign specs into
plans with stable external run ids, delegates execution to the upstream runner,
and renders reports whose claims are limited by the evidence behind them.  It
does not reimplement benchmark scheduling, sandboxing, grading, or verifiers.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
