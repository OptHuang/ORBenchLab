#!/bin/sh
set -eu
pytest --ctrf /logs/verifier/ctrf.json /tests/test_solution.py /tests/test_mutations.py -rA
