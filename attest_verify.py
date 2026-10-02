"""Smoke checks for the solver-attestation publisher surface.

The checks live in scaffold/publisher/tests/test_attest_smoke.py, where pytest
runs them with the publisher suite. This entry point keeps
``python attest_verify.py`` (the launch-readiness workflow) working.
"""
from scaffold.publisher.tests.test_attest_smoke import run_attest_smoke

if __name__ == "__main__":
    run_attest_smoke()
