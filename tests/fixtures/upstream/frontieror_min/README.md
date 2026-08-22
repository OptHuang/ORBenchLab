Miniature stand-in for a FrontierOR checkout. It carries the official entry
point's shape and the scoring-contract constants, but not a runnable package —
so the inspector's static fallback path is what these tests exercise. The real
contract read against the pinned upstream commit runs in
.github/workflows/integration-contract.yml.
