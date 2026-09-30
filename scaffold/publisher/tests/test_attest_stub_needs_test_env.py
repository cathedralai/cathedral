"""The Intel stub trusts any 632-byte blob, so it must never be the verifier of a
host that forgot to declare its environment (review plan, PR 5)."""

from __future__ import annotations

import pytest

from scaffold.publisher import attest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("CATHEDRAL_ATTEST_DCAP_VERIFY_CMD", "CATHEDRAL_DCAP_VERIFY_CMD", "CATHEDRAL_ENV",
                 "ENV", "APP_ENV", "CATHEDRAL_PRODUCTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CATHEDRAL_ATTEST_ALLOW_STUB", "1")


def test_an_undeclared_environment_never_gets_the_stub():
    assert attest.configured_intel_verifier() is None


@pytest.mark.parametrize("env", ["dev", "test", " TEST "])
def test_an_explicit_dev_or_test_environment_gets_the_stub(monkeypatch, env):
    monkeypatch.setenv("CATHEDRAL_ENV", env)
    assert isinstance(attest.configured_intel_verifier(), attest.StubIntelVerifier)


@pytest.mark.parametrize("env", ["production", "prod", "mainnet", "staging", "shadow", "development"])
def test_any_other_environment_does_not(monkeypatch, env):
    monkeypatch.setenv("CATHEDRAL_ENV", env)
    assert attest.configured_intel_verifier() is None


def test_production_flags_still_win(monkeypatch):
    monkeypatch.setenv("CATHEDRAL_ENV", "test")
    monkeypatch.setenv("CATHEDRAL_PRODUCTION", "1")
    assert attest.configured_intel_verifier() is None


def test_without_the_stub_flag_a_test_environment_gets_nothing(monkeypatch):
    monkeypatch.setenv("CATHEDRAL_ENV", "test")
    monkeypatch.delenv("CATHEDRAL_ATTEST_ALLOW_STUB")
    assert attest.configured_intel_verifier() is None


def test_an_undeclared_environment_refuses_a_632_byte_blob(tmp_path):
    from scaffold.publisher.keys import generate_test_key
    from scaffold.publisher.store import Store

    store = Store(str(tmp_path / "publisher.sqlite"))
    try:
        res = attest.verify_attestation(store, {"quote_b64": "A" * 844}, private_key_hex=generate_test_key())
    finally:
        store.close()
    assert res.ok is False and res.reason == "no_verifier_configured"


@pytest.mark.parametrize("alias", ["ENV", "APP_ENV"])
def test_any_production_alias_blocks_the_stub(monkeypatch, alias):
    monkeypatch.setenv("CATHEDRAL_ENV", "test")
    monkeypatch.setenv(alias, "production")
    assert attest.configured_intel_verifier() is None
