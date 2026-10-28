from __future__ import annotations

import os

import pytest
from pytest import MonkeyPatch

from sandbox_models.catalog import (
    LFM3_1_7B_ID,
    LFM_OPEN_LICENSE_URL,
    SandboxTier,
    UnknownSandboxModel,
    catalog_templates,
    get_model,
    list_models,
    resolve_create,
)
from sandbox_models.cpu_benchmark import parse_llama_timings
from sandbox_models.license_terms import (
    COMMERCIAL_REVENUE_THRESHOLD_USD,
    LFM_OPEN_LICENSE_SECTION_5,
    license_blocks_unrestricted_commercial_hosting,
)
from sandbox_models.sealed_test import (
    API_KEY_ENV,
    ENABLE_ENV,
    create_sealed_lfm3_body,
    run_sealed_lfm3_probe,
    sealed_test_enabled,
    skip_reason,
)


def test_lfm3_is_the_cpu_catalog_option() -> None:
    model = get_model(LFM3_1_7B_ID)
    assert list_models() == (model,)
    assert model.hf_repo == "LiquidAI/LFM3-1.7B"
    assert model.hf_gguf_repo == "LiquidAI/LFM3-1.7B-GGUF"
    assert model.gguf_filename == "LFM3-1.7B-Q4_K_M.gguf"
    assert model.runtime == "llama.cpp"
    assert model.min_runtime_version == "b11938"
    assert model.gpu_required is False
    assert model.live is False
    assert model.recommended_memory_gib == 4
    assert model.license_url == LFM_OPEN_LICENSE_URL


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(UnknownSandboxModel, match="lfm2"):
        get_model("lfm2-2.6b")


def test_standard_create_uses_catalog_template() -> None:
    body = resolve_create(LFM3_1_7B_ID, SandboxTier.STANDARD)
    assert body["template"] == LFM3_1_7B_ID
    assert body["network"] == "deny_all"
    assert "image" not in body
    assert body["labels"] == {"cathedral.model": LFM3_1_7B_ID}


def test_sealed_create_uses_custom_v1_and_workload_image() -> None:
    body = resolve_create(
        LFM3_1_7B_ID,
        SandboxTier.SEALED,
        ssh_public_key="ssh-ed25519 AAAA",
    )
    assert body["profile"] == "custom.v1"
    assert body["workload"] == {"image": "cathedral/lfm3-1.7b:q4_k_m"}
    assert body["resources"] == {
        "hardware_class": "tdx_cpu",
        "cpu": 4,
        "memory_gib": 16,
    }
    assert body["budget"]["auto_stop"] is True
    assert body["ssh_public_key"] == "ssh-ed25519 AAAA"
    lifetime = body["lifetime"]
    assert isinstance(lifetime, dict)
    assert lifetime["mode"] == "bounded_service"
    assert lifetime["reuse"] == "allowed"


def test_catalog_templates_match_standard_catalog_shape() -> None:
    templates = catalog_templates()
    assert len(templates) == 1
    entry = templates[0]
    assert entry["name"] == LFM3_1_7B_ID
    assert entry["image"] == "cathedral/lfm3-1.7b:q4_k_m"
    assert entry["resources"] == {"cpu": 1, "memory_gib": 4}
    assert entry["live"] is False
    assert "internet" in entry["network_modes"]


def test_license_has_a_ten_million_revenue_cap() -> None:
    assert COMMERCIAL_REVENUE_THRESHOLD_USD == 10_000_000
    assert license_blocks_unrestricted_commercial_hosting() is True
    assert "not exceeding the Threshold" in LFM_OPEN_LICENSE_SECTION_5
    assert "exceeds the Threshold is not licensed" in LFM_OPEN_LICENSE_SECTION_5


def test_parse_llama_timings_from_cli_stderr() -> None:
    stderr = """
llama_model_loader: loaded meta data with 42 key-value pairs
llama_perf_context_print:        load time =    1842.10 ms
llama_perf_context_print: prompt eval time =     210.00 ms /    12 tokens
llama_perf_context_print:        eval time =    1600.00 ms /    32 tokens
[ Prompt: 81.5 t/s | Generation: 18.5 t/s ]
"""
    timings = parse_llama_timings(stderr)
    assert timings["load_ms"] == 1842.10
    assert timings["prompt_eval_ms"] == 210.00
    assert timings["prompt_tokens"] == 12
    assert timings["eval_ms"] == 1600.00
    assert timings["eval_tokens"] == 32
    assert timings["prompt_tps"] == 81.5
    assert timings["generation_tps"] == 18.5


def test_sealed_probe_skips_without_explicit_enable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert sealed_test_enabled() is False
    assert ENABLE_ENV in skip_reason()


def test_sealed_probe_stays_off_when_only_a_key_is_present(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "not-a-production-key")
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert sealed_test_enabled() is False


def test_sealed_probe_body_is_custom_v1(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("CATHEDRAL_SSH_PUBLIC_KEY", "ssh-ed25519 AAAA")
    body = create_sealed_lfm3_body()
    assert body["profile"] == "custom.v1"
    workload = body["workload"]
    assert isinstance(workload, dict)
    assert str(workload["image"]).endswith("lfm3-1.7b:q4_k_m")
    assert str(body["name"]).startswith(LFM3_1_7B_ID)


def test_live_sealed_run_is_opt_in() -> None:
    if os.environ.get(ENABLE_ENV) == "1" and os.environ.get(API_KEY_ENV):
        report = run_sealed_lfm3_probe()
        assert report["host"] == "sealed"
        assert report["boot_to_first_token_seconds"] is not None
        return
    assert sealed_test_enabled() is False
