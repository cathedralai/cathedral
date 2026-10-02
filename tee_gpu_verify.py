"""Smoke checks for the TEE GPU capacity lane.

The checks live in scaffold/publisher/tests/test_tee_gpu_smoke.py, where pytest
runs them with the publisher suite. This entry point keeps
``python tee_gpu_verify.py`` (the launch-readiness workflow) working.
"""
from scaffold.publisher.tests.test_tee_gpu_smoke import run_tee_gpu_smoke

if __name__ == "__main__":
    run_tee_gpu_smoke()
