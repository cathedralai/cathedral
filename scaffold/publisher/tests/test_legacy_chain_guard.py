"""Legacy scaffold chain paths must never broadcast on Finney (mainnet)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


FINNEY_GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
TESTNET_GENESIS = "0x" + "ab" * 32
CONFIG = (
    Path(__file__).resolve().parents[3] / "config" / "validator-thin-sn39-relay.toml"
)


@pytest.mark.parametrize(
    "network",
    ["finney", "archive", "wss://entrypoint-finney.opentensor.ai:443", ""],
)
def test_legacy_chain_client_refuses_finney_broadcast_on_any_netuid(
    network: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ChainClient(netuid=94, network=network, broadcast=True)
    monkeypatch.setattr(
        client,
        "_bittensor",
        lambda: (_ for _ in ()).throw(
            AssertionError("a Finney refusal must happen before chain setup")
        ),
    )
    result = client.set_weights(WeightVector(by_uid={1: 1.0}))
    assert result["submitted"] is False
    assert "disabled on Finney" in result["reason"]


def _fake_chain(genesis):
    calls: list[dict] = []

    def get_block_hash(block):
        assert block == 0
        if isinstance(genesis, Exception):
            raise genesis
        return genesis

    subtensor = SimpleNamespace(
        substrate=SimpleNamespace(get_block_hash=get_block_hash),
        set_weights=lambda **kwargs: calls.append(kwargs) or (True, "ok"),
    )
    return subtensor, calls


@pytest.mark.parametrize(
    "genesis", [FINNEY_GENESIS, FINNEY_GENESIS.upper(), None, "0x12", OSError("rpc")]
)
def test_legacy_chain_client_refuses_a_test_label_on_a_finney_or_unknown_chain(
    genesis: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ChainClient(netuid=94, network="test", broadcast=True)
    subtensor, calls = _fake_chain(genesis)
    monkeypatch.setattr(client, "_bittensor", lambda: object())
    monkeypatch.setattr(client, "_subtensor", lambda _bt: subtensor)
    monkeypatch.setattr(client, "_wallet", lambda _bt: object())
    result = client.set_weights(WeightVector(by_uid={1: 1.0}))
    assert result["submitted"] is False
    assert "disabled on Finney" in result["reason"]
    assert calls == []


def test_legacy_chain_client_still_broadcasts_on_a_proven_testnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ChainClient(netuid=94, network="test", broadcast=True)
    subtensor, calls = _fake_chain(TESTNET_GENESIS)
    monkeypatch.setattr(client, "_bittensor", lambda: object())
    monkeypatch.setattr(client, "_subtensor", lambda _bt: subtensor)
    monkeypatch.setattr(client, "_wallet", lambda _bt: object())
    result = client.set_weights(WeightVector(by_uid={1: 1.0}))
    assert result["submitted"] is True
    assert [call["netuid"] for call in calls] == [94]


def test_legacy_chain_client_dry_run_needs_no_chain() -> None:
    client = ChainClient(netuid=94, network="finney", broadcast=False)
    result = client.set_weights(WeightVector(by_uid={1: 1.0}))
    assert result["submitted"] is False
    assert result["dry_run"] is True


def _fail_if_run(*_args, **_kwargs):
    raise AssertionError("a Finney broadcast must be refused before the writer runs")


@pytest.fixture(autouse=True)
def _no_ambient_cathedral_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The serve config reads CATHEDRAL_* overrides from the environment.
    import os

    for name in list(os.environ):
        if name.startswith("CATHEDRAL_"):
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("netuid", ["39", "94", "2"])
def test_cli_serve_refuses_finney_broadcast_on_any_netuid(
    netuid: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scaffold import cli, validator_thin

    monkeypatch.setattr(validator_thin, "run", _fail_if_run)
    argv = ["serve", "--config", str(CONFIG), "--netuid", netuid, "--broadcast"]
    assert cli.main(argv) == 2
    assert "does not broadcast weights on Finney" in capsys.readouterr().err


def test_cli_serve_refuses_an_unreadable_netuid(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scaffold import cli, validator_thin

    monkeypatch.setattr(validator_thin, "run", _fail_if_run)
    # The flag is int-typed; the environment override is read as text.
    monkeypatch.setenv("CATHEDRAL_WEIGHT_POLICY_NETUID", "ninety-four")
    assert cli.main(["serve", "--config", str(CONFIG), "--broadcast"]) == 2
    assert "invalid serve configuration" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--config", str(CONFIG), "--netuid", "x", "--broadcast"])
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["--dry-run", "--offline"])
def test_cli_serve_dry_run_and_offline_still_reach_the_writer(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scaffold import cli, validator_thin

    seen = {}

    def fake_run(cfg):
        seen["broadcast"] = cfg.broadcast
        seen["netuid"] = cfg.netuid
        return 0

    monkeypatch.setattr(validator_thin, "run", fake_run)
    argv = ["serve", "--config", str(CONFIG), "--netuid", "94", mode, "--broadcast"]
    assert cli.main(argv) == 0
    assert seen == {"broadcast": False, "netuid": 94}


def _main_argv(*extra: str) -> list[str]:
    return ["validator_thin", "--public-key-hex", "00" * 32, *extra]


@pytest.mark.parametrize(
    "extra",
    [
        ("--netuid", "39", "--broadcast"),
        ("--netuid", "94", "--broadcast"),
        ("--network", "finney", "--netuid", "94", "--broadcast"),
        ("--network", "test", "--netuid", "39", "--broadcast"),
    ],
)
def test_validator_thin_main_refuses_finney_or_sn39_broadcast(
    extra: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scaffold import validator_thin

    monkeypatch.setattr(validator_thin, "run", _fail_if_run)
    monkeypatch.setattr("sys.argv", _main_argv(*extra))
    with pytest.raises(SystemExit) as exc:
        validator_thin.main()
    assert exc.value.code == 2
    assert "does not broadcast weights on Finney" in capsys.readouterr().err


def test_validator_thin_main_refuses_an_unreadable_netuid(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scaffold import validator_thin

    monkeypatch.setattr(validator_thin, "run", _fail_if_run)
    monkeypatch.setattr("sys.argv", _main_argv("--netuid", "x94", "--broadcast"))
    with pytest.raises(SystemExit) as exc:
        validator_thin.main()
    assert exc.value.code == 2
    assert "--netuid" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [(), ("--offline", "--broadcast")])
def test_validator_thin_main_dry_run_and_offline_still_run(
    extra: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scaffold import validator_thin

    seen = {}

    def fake_run(args):
        seen["netuid"] = args.netuid
        return 0

    monkeypatch.setattr(validator_thin, "run", fake_run)
    monkeypatch.setattr("sys.argv", _main_argv("--netuid", "94", *extra))
    assert validator_thin.main() == 0
    assert seen == {"netuid": 94}


@pytest.mark.parametrize(
    ("args", "refused"),
    [
        ({"broadcast": True, "netuid": 39, "network": "finney"}, True),
        ({"broadcast": True, "netuid": 94, "network": "finney"}, True),
        ({"broadcast": True, "netuid": "94", "network": " Finney "}, True),
        ({"broadcast": True, "netuid": 94, "network": "archive"}, True),
        ({"broadcast": True, "netuid": 94, "network": "wss://node:443"}, True),
        ({"broadcast": True, "netuid": 94}, True),
        ({"broadcast": True, "netuid": 39, "network": "test"}, True),
        ({"broadcast": True, "netuid": None, "network": "test"}, True),
        ({"broadcast": True, "netuid": "94x", "network": "test"}, True),
        ({"broadcast": True, "netuid": 9.4, "network": "test"}, True),
        ({"broadcast": True, "netuid": True, "network": "test"}, True),
        ({"broadcast": True, "netuid": -1, "network": "test"}, True),
        ({"broadcast": True, "network": "test"}, True),
        ({"broadcast": True, "netuid": 94, "network": "test"}, False),
        ({"broadcast": True, "netuid": "2", "network": "local"}, False),
        ({"broadcast": False, "netuid": 94, "network": "finney"}, False),
        ({"broadcast": False, "netuid": "x", "network": "finney"}, False),
        (
            {"broadcast": True, "offline": True, "netuid": 94, "network": "finney"},
            False,
        ),
    ],
)
def test_legacy_broadcast_refusal_scope(args: dict, refused: bool) -> None:
    from scaffold import validator_thin

    result = validator_thin.legacy_broadcast_refusal(SimpleNamespace(**args))
    assert (result is not None) is refused


def _resolved_preflight(genesis: str):
    from scaffold import validator_thin

    return validator_thin.ChainPreflight(
        wallet=object(),
        subtensor=object(),
        hotkey_to_uid={},
        validator_hotkey="validator-hotkey",
        validator_uid=0,
        block=1,
        min_allowed_weights=1,
        max_weight_limit=1.0,
        genesis_hash=genesis,
    )


@pytest.mark.parametrize("netuid", [94, 2])
def test_connected_finney_chain_refuses_a_test_label_broadcast(netuid: int) -> None:
    from scaffold import validator_thin

    args = SimpleNamespace(broadcast=True, offline=False, netuid=netuid, network="test")
    with pytest.raises(validator_thin.wire.VectorError, match="on Finney"):
        validator_thin._validate_resolved_chain_contract(
            args, _resolved_preflight(FINNEY_GENESIS)
        )


def test_connected_testnet_chain_allows_a_test_label_broadcast() -> None:
    from scaffold import validator_thin

    args = SimpleNamespace(broadcast=True, offline=False, netuid=94, network="test")
    validator_thin._validate_resolved_chain_contract(
        args, _resolved_preflight(TESTNET_GENESIS)
    )
    dry_run = SimpleNamespace(
        broadcast=False, offline=False, netuid=94, network="finney"
    )
    validator_thin._validate_resolved_chain_contract(
        dry_run, _resolved_preflight(FINNEY_GENESIS)
    )
