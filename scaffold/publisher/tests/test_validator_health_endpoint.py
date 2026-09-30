"""Integration tests for the read-only validator-health endpoint + 5xx counter.

Uses the in-process app (sqlite :memory:) via TestClient. No network, no chain.
"""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from scaffold.publisher import build_app
from scaffold.publisher.keys import generate_test_key

ADMIN_TOKEN = "test-admin-token"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CATHEDRAL_PUBLISHER_ADMIN_TOKEN", ADMIN_TOKEN)
    # service_role defaults to "all" so the admin route is reachable.
    app = build_app(database_path=":memory:", signing_key_hex=generate_test_key())
    with TestClient(app) as c:
        yield c


def test_validator_health_requires_admin_token(client):
    # No bearer -> 401 (token configured) per _require_publisher_admin.
    resp = client.get("/v1/admin/validator-health")
    assert resp.status_code == 401


def test_validator_health_wrong_token_rejected(client):
    resp = client.get(
        "/v1/admin/validator-health",
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_validator_health_shape(client):
    resp = client.get(
        "/v1/admin/validator-health",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "no-store"
    body = resp.json()
    assert body["schema"] == "cathedral.validator_health.v1"
    assert "weights_feed" in body
    assert "freshness" in body["weights_feed"]
    assert "level" in body["weights_feed"]["freshness"]
    assert body["weights_feed"]["freshness"]["hard_ceiling_seconds"] == 4320
    assert "http_status" in body
    assert "by_class" in body["http_status"]
    assert "rate_5xx" in body["http_status"]
    assert "submit" in body
    assert body["tempo_seconds"] == 4320


def test_4xx_counted_but_404_is_not_a_5xx(client):
    # A 404 (unknown route) is a 4xx, not a server fault: confirm 4xx is counted
    # and the 5xx class is left untouched. The genuine 5xx path is covered by
    # test_unhandled_exception_counts_as_5xx below.
    before = client.get(
        "/v1/admin/validator-health",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json()["http_status"]
    # Trigger a 404 (unknown route) -> counted as 4xx, not 5xx.
    client.get("/v1/this-route-does-not-exist")
    after = client.get(
        "/v1/admin/validator-health",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json()["http_status"]
    assert after["by_class"]["4xx"] > before["by_class"]["4xx"]
    # No server faults were induced.
    assert after["by_class"]["5xx"] == before["by_class"]["5xx"]


def test_weights_feed_freshness_present_after_feed_hit(client):
    # Warm the weight feed so cached_vector has a vector to peek.
    feed = client.get("/v1/validator/weights/next")
    assert feed.status_code == 200
    assert "generated_at" in feed.json()

    body = client.get(
        "/v1/admin/validator-health",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    ).json()
    wf = body["weights_feed"]
    assert wf["vector_present"] is True
    assert wf["generated_at"] is not None
    # freshly generated -> age should be tiny and classified ok.
    assert wf["freshness"]["age_seconds"] is not None
    assert wf["freshness"]["level"] in {"ok", "warn"}
    # the feed request itself was a 2xx, so feed 5xx stays 0.
    assert wf["feed_5xx"] == 0


def test_unhandled_exception_counts_as_5xx(monkeypatch):
    # The whole point of _StatusCounterMiddleware: an unhandled 500 (the route
    # raises and never sends a response-start) must still be tallied. We inject a
    # raising route on a fresh app and use raise_server_exceptions=False so the
    # TestClient surfaces the 500 instead of re-raising the exception.
    monkeypatch.setenv("CATHEDRAL_PUBLISHER_ADMIN_TOKEN", ADMIN_TOKEN)
    app = build_app(database_path=":memory:", signing_key_hex=generate_test_key())

    @app.get("/v1/test-only/boom")
    async def _boom():
        raise RuntimeError("synthetic unhandled error")

    with TestClient(app, raise_server_exceptions=False) as c:
        before = c.get(
            "/v1/admin/validator-health",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        ).json()["http_status"]

        resp = c.get("/v1/test-only/boom")
        assert resp.status_code == 500

        after = c.get(
            "/v1/admin/validator-health",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        ).json()["http_status"]

    # The synthetic 500 must be reflected in both the per-class tally and the
    # recent-5xx ring buffer.
    assert after["by_class"]["5xx"] == before["by_class"]["5xx"] + 1
    assert len(after["recent_5xx"]) == len(before["recent_5xx"]) + 1
    assert after["recent_5xx"][-1]["status"] == 500
    assert after["recent_5xx"][-1]["path"] == "/v1/test-only/boom"


def _health_with_vector(monkeypatch, client, vec):
    from scaffold.publisher import weights as weights_mod

    monkeypatch.setattr(weights_mod, "cached_vector", lambda *a, **k: vec)
    resp = client.get(
        "/v1/admin/validator-health",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert resp.status_code == 200
    return resp.json()["composition"]


def test_composition_block_without_a_vector(monkeypatch, client):
    monkeypatch.setenv("CATHEDRAL_WEIGHTS_COLDKEY_COLLAPSE", "1")
    block = _health_with_vector(monkeypatch, client, None)  # a cold process: nothing cached
    assert block["vector_present"] is False and block["alerts"] == []


def test_composition_block_flags_the_silent_shifts_of_issue_345(monkeypatch, client):
    monkeypatch.setenv("CATHEDRAL_WEIGHTS_COLDKEY_COLLAPSE", "1")
    vec = {
        "generated_at": "2026-01-01T00:00:00.000Z",
        "policy_metadata": {
            "miner_count": 12,
            "coldkey_map_loaded": False,
            "external_scores": {"enabled": True, "has_scores": False},
            "payable_hotkeys": {
                "mode": "filter",
                "status": "no_fresh_snapshot",
                "snapshot_fresh": False,
            },
        },
    }
    block = _health_with_vector(monkeypatch, client, vec)
    assert block["miner_count"] == 12
    assert block["payable_policy_status"] == "no_fresh_snapshot"
    assert block["metagraph_snapshot_fresh"] is False
    assert block["alerts"] == ["external_scores_empty", "metagraph_snapshot_stale", "coldkey_map_missing"]


def test_composition_block_is_quiet_when_healthy(monkeypatch, client):
    monkeypatch.delenv("CATHEDRAL_WEIGHTS_COLDKEY_COLLAPSE", raising=False)
    vec = {
        "generated_at": "2026-01-01T00:00:00.000Z",
        "policy_metadata": {
            "miner_count": 5,
            "coldkey_map_loaded": False,
            "external_scores": {"enabled": True, "has_scores": True},
            "payable_hotkeys": {"mode": "filter", "status": "all_payable", "snapshot_fresh": True},
        },
    }
    block = _health_with_vector(monkeypatch, client, vec)
    assert block["external_has_scores"] is True and block["alerts"] == []


def test_composition_block_reads_a_real_vector(monkeypatch, client, tmp_path):
    """Keys come from build_signed_vector's own metadata: a payable filter with no
    metagraph snapshot (an empty store) shows up as stale."""
    from scaffold.publisher import weights as weights_mod
    from scaffold.publisher.store import Store

    monkeypatch.setenv("CATHEDRAL_WEIGHTS_PAYABLE_HOTKEYS", "filter")
    vec = weights_mod.build_signed_vector(
        Store(str(tmp_path / "publisher.sqlite")), signing_key_hex="11" * 32
    )
    block = _health_with_vector(monkeypatch, client, vec)
    assert block["vector_present"] is True
    assert block["payable_policy_mode"] == "filter"
    assert block["payable_policy_status"] == "no_fresh_snapshot"
    assert block["metagraph_snapshot_fresh"] is False
    assert "metagraph_snapshot_stale" in block["alerts"]


def test_composition_block_tolerates_malformed_metadata(monkeypatch, client):
    vec = {"policy_metadata": {"payable_hotkeys": "junk", "external_scores": ["junk"]}}
    block = _health_with_vector(monkeypatch, client, vec)
    assert block["payable_policy_status"] is None and block["alerts"] == []
