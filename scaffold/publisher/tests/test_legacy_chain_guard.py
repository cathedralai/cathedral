"""Legacy scaffold chain paths must never become a second SN39 writer."""

from __future__ import annotations

import pytest

from scaffold.chain import ChainClient, WeightVector


@pytest.mark.parametrize(
    "network",
    [
        "finney",
        "test",
        "wss://entrypoint-finney.opentensor.ai:443",
        "wss://self-hosted-finney.example",
    ],
)
def test_legacy_chain_client_refuses_sn39_before_importing_bittensor(
    network: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ChainClient(netuid=39, network=network, broadcast=True)
    monkeypatch.setattr(
        client,
        "_bittensor",
        lambda: (_ for _ in ()).throw(
            AssertionError("SN39 refusal must happen before chain setup")
        ),
    )
    result = client.set_weights(WeightVector(by_uid={1: 1.0}))
    assert result["submitted"] is False
    assert "disabled on SN39" in result["reason"]


def _fail_if_run(*_args, **_kwargs):
    raise AssertionError("an SN39 broadcast must be refused before the writer runs")


@pytest.fixture(autouse=True)
def _no_ambient_cathedral_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The serve config reads CATHEDRAL_* overrides from the environment.
    import os

    for name in list(os.environ):
        if name.startswith("CATHEDRAL_"):
            monkeypatch.delenv(name, raising=False)


def test_cli_serve_refuses_sn39_broadcast_before_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pathlib import Path

    from scaffold import cli, validator_thin

    monkeypatch.setattr(validator_thin, "run", _fail_if_run)
    config = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "validator-thin-sn39-relay.toml"
    )
    assert cli.main(["serve", "--config", str(config), "--broadcast"]) == 2
    assert "does not write SN39 weights" in capsys.readouterr().err


def test_cli_serve_dry_run_still_reaches_the_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from scaffold import cli, validator_thin

    seen = {}

    def fake_run(cfg):
        seen["broadcast"] = cfg.broadcast
        return 0

    monkeypatch.setattr(validator_thin, "run", fake_run)
    config = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "validator-thin-sn39-relay.toml"
    )
    assert cli.main(["serve", "--config", str(config), "--dry-run"]) == 0
    assert seen == {"broadcast": False}


def test_validator_thin_main_refuses_sn39_broadcast(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scaffold import validator_thin

    monkeypatch.setattr(validator_thin, "run", _fail_if_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validator_thin",
            "--public-key-hex",
            "00" * 32,
            "--netuid",
            "39",
            "--broadcast",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        validator_thin.main()
    assert exc.value.code == 2
    assert "does not write SN39 weights" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("broadcast", "netuid", "refused"),
    [(True, 39, True), (True, "39", True), (False, 39, False), (True, 2, False)],
)
def test_legacy_sn39_broadcast_refusal_scope(
    broadcast: bool, netuid: object, refused: bool
) -> None:
    from types import SimpleNamespace

    from scaffold import validator_thin

    result = validator_thin.legacy_sn39_broadcast_refusal(
        SimpleNamespace(broadcast=broadcast, netuid=netuid)
    )
    assert (result is not None) is refused
