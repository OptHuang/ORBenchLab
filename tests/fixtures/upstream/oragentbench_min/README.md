Miniature stand-in for an ORAgentBench checkout, used by the offline integration
tests. It reproduces the shape the inspector reads — Harbor task packages, the
reward channel, the official metric script and the oracle job config — and
nothing else. The real contract check against the pinned upstream commit runs in
.github/workflows/integration-contract.yml.
