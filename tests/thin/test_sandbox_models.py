from __future__ import annotations

import os
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from sandbox_models.catalog import (
    D1_2B_ID,
    D1_OMNI_2B_ID,
    LFM3_1_7B_BASE_ID,
    LFM3_1_7B_ID,
    LFM3_AUDIO_2B_ID,
    LFM3_OMNI_2B_ID,
    LFM3_VL_2B_ID,
    LFM_OPEN_LICENSE_URL,
    LFM_OPEN_LICENSE_VERSION,
    SandboxTier,
    UnknownSandboxModel,
    catalog_templates,
    get_model,
    list_models,
    resolve_create,
)
from sandbox_models.cpu_benchmark import (
    parse_llama_timings,
    write_red_png,
    write_sine_wav,
)
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


SUITE_IDS = (
    LFM3_1_7B_ID,
    LFM3_1_7B_BASE_ID,
    LFM3_VL_2B_ID,
    LFM3_AUDIO_2B_ID,
    LFM3_OMNI_2B_ID,
    D1_2B_ID,
    D1_OMNI_2B_ID,
)


def test_lfm3_1_7b_catalog_fields() -> None:
    model = get_model(LFM3_1_7B_ID)
    assert model.hf_repo == "LiquidAI/LFM3-1.7B"
    assert model.hf_gguf_repo == "LiquidAI/LFM3-1.7B-GGUF"
    assert model.gguf_filename == "LFM3-1.7B-Q4_K_M.gguf"
    assert model.runtime == "llama.cpp"
    assert model.min_runtime_version == "b11938"
    assert model.gpu_required is False
    assert model.live is False
    assert model.recommended_memory_gib == 4
    assert model.license_url == LFM_OPEN_LICENSE_URL
    assert "vLLM" in model.runtimes
    assert "transformers" in model.runtimes


def test_suite_ids_are_registered() -> None:
    ids = {model.id for model in list_models()}
    assert ids == set(SUITE_IDS)
    assert len(list_models()) == 7


@pytest.mark.parametrize("model_id", SUITE_IDS)
def test_each_published_model_uses_lfm_open_license(model_id: str) -> None:
    model = get_model(model_id)
    assert model.license_id == LFM_OPEN_LICENSE_VERSION
    assert model.live is False
    assert model.gpu_required is False


def test_base_has_no_published_gguf() -> None:
    model = get_model(LFM3_1_7B_BASE_ID)
    assert model.hf_repo == "LiquidAI/LFM3-1.7B-Base"
    assert model.hf_gguf_repo == ""
    assert model.gguf_filename == ""
    assert model.runtime == "transformers"
    assert model.weight_format == "safetensors"
    assert model.weight_bytes == 3_457_635_448


def test_vl_sandbox_drops_video() -> None:
    model = get_model(LFM3_VL_2B_ID)
    assert model.hf_repo == "LiquidAI/LFM3-VL-2B"
    assert model.hf_gguf_repo == "LiquidAI/LFM3-VL-2B-GGUF"
    assert model.gguf_filename == "LFM3-VL-2B-Q4_K_M.gguf"
    assert model.mmproj_filename == "mmproj-LFM3-VL-2B-Q8_0.gguf"
    assert model.mmproj_bytes == 203_031_872
    assert model.input_modalities == ("text", "image", "video")
    assert model.sandbox_input_modalities == ("text", "image")
    assert model.sandbox_output_modalities == ("text",)
    assert "video" not in model.sandbox_input_modalities


def test_audio_and_omni_need_llama_cpp_b11941() -> None:
    audio = get_model(LFM3_AUDIO_2B_ID)
    omni = get_model(LFM3_OMNI_2B_ID)
    assert audio.min_runtime_version == "b11941"
    assert omni.min_runtime_version == "b11941"
    assert audio.sandbox_input_modalities == ("text", "audio")
    assert audio.sandbox_output_modalities == ("text", "speech")
    assert omni.sandbox_input_modalities == ("text", "image", "audio")
    assert omni.sandbox_output_modalities == ("text", "speech")
    assert "video" not in omni.sandbox_input_modalities


def test_d1_models_use_transformers_until_llama_cpp_lands() -> None:
    d1 = get_model(D1_2B_ID)
    omni = get_model(D1_OMNI_2B_ID)
    assert d1.runtime == "transformers"
    assert omni.runtime == "transformers"
    assert d1.hf_gguf_repo == "LiquidAI/d1-2B-GGUF"
    assert omni.hf_gguf_repo == "LiquidAI/d1-Omni-2B-GGUF"
    assert d1.gguf_filename == "d1-2B-Q4_K_M.gguf"
    assert omni.gguf_filename == "d1-Omni-2B-Q8_0.gguf"
    assert "31275" in d1.sandbox_notes
    assert "31275" in omni.sandbox_notes
    assert d1.sandbox_output_modalities == ("decision",)
    assert omni.sandbox_output_modalities == ("decision",)


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(UnknownSandboxModel, match="lfm2"):
        get_model("lfm2-2.6b")


@pytest.mark.parametrize("model_id", SUITE_IDS)
def test_standard_create_uses_catalog_template(model_id: str) -> None:
    body = resolve_create(model_id, SandboxTier.STANDARD)
    assert body["template"] == model_id
    assert body["network"] == "deny_all"
    assert "image" not in body
    assert body["labels"] == {"cathedral.model": model_id}


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


def test_sealed_memory_floors_at_sixteen_gib() -> None:
    body = resolve_create(D1_OMNI_2B_ID, SandboxTier.SEALED)
    resources = body["resources"]
    assert isinstance(resources, dict)
    assert resources["memory_gib"] == 16
    assert resources["hardware_class"] == "tdx_cpu"


def test_catalog_templates_match_standard_catalog_shape() -> None:
    templates = catalog_templates()
    assert len(templates) == 7
    by_name = {entry["name"]: entry for entry in templates}
    entry = by_name[LFM3_1_7B_ID]
    assert entry["image"] == "cathedral/lfm3-1.7b:q4_k_m"
    assert entry["resources"] == {"cpu": 1, "memory_gib": 4}
    assert entry["live"] is False
    assert "internet" in entry["network_modes"]
    assert "llama.cpp" in str(entry["description"])
    vl = by_name[LFM3_VL_2B_ID]
    assert vl["input_modalities"] == ["text", "image"]
    assert "video" not in vl["input_modalities"]


def test_each_model_has_an_image_recipe() -> None:
    root = Path(__file__).resolve().parents[2] / "images"
    for model in list_models():
        image_dir = root / model.id
        assert (image_dir / "Dockerfile").is_file(), model.id
        assert (image_dir / "LICENSE").is_file(), model.id


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


def test_smoke_media_helpers(tmp_path: Path) -> None:
    png = write_red_png(tmp_path / "red.png")
    wav = write_sine_wav(tmp_path / "tone.wav")
    assert png.stat().st_size > 20
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert wav.read_bytes()[:4] == b"RIFF"


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
