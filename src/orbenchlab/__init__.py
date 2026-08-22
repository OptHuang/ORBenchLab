"""ORBenchLab — a control plane for operations-research agent benchmarks.

ORBenchLab does not execute benchmarks and does not reimplement them. It
registers integrations with upstream benchmarks, compiles campaign specs into
plans with stable external run ids, and renders reports whose claims are limited
by the evidence behind them.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
