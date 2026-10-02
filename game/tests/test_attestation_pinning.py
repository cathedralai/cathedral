"""Attestation pins the image by content digest; a bare (mutable) tag is refused,
because whoever controls the tag chooses what the box runs."""

from __future__ import annotations

from scaffold.polaris import PolarisClient, offline_pinned_ref
from scaffold.verify import verify_attestation

PUBKEY = "cHVi"


def _attest(image: str):
    ok, res = verify_attestation(PolarisClient(), nonce="n-1", pubkey_b64=PUBKEY,
                                 expected_image=image, workload="solve")
    return ok, res


def test_a_digest_pinned_image_attests():
    ok, res = _attest(offline_pinned_ref("goodsolver"))
    assert ok is True and res.image_digest.endswith(offline_pinned_ref("goodsolver")[-64:])


def test_a_bare_tag_is_refused():
    ok, res = _attest("registry.example/goodsolver:latest")
    assert ok is False
    assert res.intel_verified is True       # the quote itself was fine: the pin is what failed


def test_a_different_digest_is_refused(monkeypatch):
    client = PolarisClient()
    real = client.attest

    def ran_another_image(**kwargs):
        return real(**{**kwargs, "image": offline_pinned_ref("other")})

    monkeypatch.setattr(client, "attest", ran_another_image)
    ok, _ = verify_attestation(client, nonce="n-1", pubkey_b64=PUBKEY,
                               expected_image=offline_pinned_ref("goodsolver"), workload="solve")
    assert ok is False


def test_arena_server_binds_loopback_unless_told_otherwise(monkeypatch, tmp_path):
    from game.arena import serve as srv

    bound = []

    class FakeServer:
        def __init__(self, address, handler):
            bound.append(address)

        def serve_forever(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(srv, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(srv, "OUT", tmp_path)
    monkeypatch.setattr(srv, "ArenaServer", lambda **kw: object())
    monkeypatch.setattr(srv, "_handler", lambda s: None)
    monkeypatch.delenv("CATHEDRAL_ARENA_HOST", raising=False)
    srv.serve(8801)
    monkeypatch.setenv("CATHEDRAL_ARENA_HOST", "0.0.0.0")
    srv.serve(8802)
    assert bound == [("127.0.0.1", 8801), ("0.0.0.0", 8802)]


def test_hex_in_an_image_name_is_not_a_pin():
    ok, _ = _attest("evil/" + "a" * 64 + ":latest")
    assert ok is False
