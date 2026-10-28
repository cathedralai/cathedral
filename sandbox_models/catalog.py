"""Catalog of models a caller can pick when creating a Cathedral sandbox.

The live Standard catalog is ``GET /v1/sandboxes/catalog``. The live sealed
catalog is ``GET /v1/profiles`` (``custom.v1``). This module is the source of
truth in this repository for the LFM3 suite. The hotel broker in
cathedral-sandbox and the private service layer must consume the same
template name and image before an option is live.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, assert_never


LFM3_1_7B_ID = "lfm3-1.7b"
LFM3_1_7B_BASE_ID = "lfm3-1.7b-base"
LFM3_VL_2B_ID = "lfm3-vl-2b"
LFM3_AUDIO_2B_ID = "lfm3-audio-2b"
LFM3_OMNI_2B_ID = "lfm3-omni-2b"
D1_2B_ID = "d1-2b"
D1_OMNI_2B_ID = "d1-omni-2b"

LFM_OPEN_LICENSE_VERSION = "LFM Open License v1.0"
LFM_OPEN_LICENSE_URL = "https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE"
LFM_OPEN_LICENSE_GUIDE_URL = "https://docs.liquid.ai/lfm/help/model-license"

# lfm3-1.7b is published on GHCR (Spock). Other suite images still local-only.
LFM3_IMAGE_REF = "ghcr.io/cathedralai/room-lfm3-1.7b@sha256:b1ba8837ea388c8c98720772b0f41af5813d4cebf214637fde20aae58a1ea761"
LFM3_BASE_IMAGE_REF = "cathedral/lfm3-1.7b-base:bf16"
LFM3_VL_IMAGE_REF = "cathedral/lfm3-vl-2b:q4_k_m"
LFM3_AUDIO_IMAGE_REF = "cathedral/lfm3-audio-2b:q4_k_m"
LFM3_OMNI_IMAGE_REF = "cathedral/lfm3-omni-2b:q4_k_m"
D1_2B_IMAGE_REF = "cathedral/d1-2b:bf16"
D1_OMNI_IMAGE_REF = "cathedral/d1-omni-2b:bf16"

LFM3_GGUF_REPO = "LiquidAI/LFM3-1.7B-GGUF"
LFM3_GGUF_FILENAME = "LFM3-1.7B-Q4_K_M.gguf"
LFM3_WEIGHT_BYTES = 1_099_627_808
LFM3_MIN_LLAMA_CPP = "b11938"
LFM3_AUDIO_MIN_LLAMA_CPP = "b11941"
D1_MIN_TRANSFORMERS = "5.19"

# Hugging Face returned 401 for LiquidAI/LFM3-1.7B-Base-GGUF. Not published.
LFM3_1_7B_BASE_GGUF_REPO = ""


class SandboxTier(Enum):
    STANDARD = "standard"
    SEALED = "sealed"


class UnknownSandboxModel(KeyError):
    """The requested model id is not in this catalog."""


@dataclass(frozen=True)
class SandboxModel:
    id: str
    display_name: str
    family: str
    hf_repo: str
    hf_gguf_repo: str
    gguf_filename: str
    runtime: str
    min_runtime_version: str
    parameters: str
    weight_bytes: int
    recommended_vcpu: int
    recommended_memory_gib: int
    recommended_disk_gib: int
    standard_template: str
    sealed_image: str
    license_id: str
    license_url: str
    gpu_required: bool
    live: bool
    mmproj_filename: str = ""
    mmproj_bytes: int = 0
    weight_format: str = "gguf"
    runtimes: tuple[str, ...] = ("llama.cpp",)
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    sandbox_input_modalities: tuple[str, ...] = ("text",)
    sandbox_output_modalities: tuple[str, ...] = ("text",)
    sandbox_notes: str = ""


def _license_url(hf_repo: str) -> str:
    return f"https://huggingface.co/{hf_repo}/blob/main/LICENSE"


_LFM3_1_7B = SandboxModel(
    id=LFM3_1_7B_ID,
    display_name="LFM3 1.7B (CPU)",
    family="LFM3",
    hf_repo="LiquidAI/LFM3-1.7B",
    hf_gguf_repo=LFM3_GGUF_REPO,
    gguf_filename=LFM3_GGUF_FILENAME,
    runtime="llama.cpp",
    min_runtime_version=LFM3_MIN_LLAMA_CPP,
    parameters="1.73B",
    weight_bytes=LFM3_WEIGHT_BYTES,
    recommended_vcpu=2,
    recommended_memory_gib=8,
    recommended_disk_gib=10,
    standard_template=LFM3_1_7B_ID,
    sealed_image=LFM3_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=LFM_OPEN_LICENSE_URL,
    gpu_required=False,
    live=True,
    weight_format="gguf",
    runtimes=("llama.cpp", "vLLM", "transformers"),
    sandbox_notes="Standard self-serve on broker (GHCR pin). Sealed pending hotel pull timeout.",
)

_LFM3_1_7B_BASE = SandboxModel(
    id=LFM3_1_7B_BASE_ID,
    display_name="LFM3 1.7B Base (CPU)",
    family="LFM3",
    hf_repo="LiquidAI/LFM3-1.7B-Base",
    hf_gguf_repo=LFM3_1_7B_BASE_GGUF_REPO,
    gguf_filename="",
    runtime="transformers",
    min_runtime_version="4.0",
    parameters="1.73B",
    weight_bytes=3_457_635_448,
    recommended_vcpu=2,
    recommended_memory_gib=8,
    recommended_disk_gib=16,
    standard_template=LFM3_1_7B_BASE_ID,
    sealed_image=LFM3_BASE_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=_license_url("LiquidAI/LFM3-1.7B-Base"),
    gpu_required=False,
    live=False,
    weight_format="safetensors",
    runtimes=("transformers", "vLLM"),
    sandbox_notes=(
        "No public GGUF. LiquidAI/LFM3-1.7B-Base-GGUF is not published "
        "(Hugging Face 401). Pretrain checkpoint for fine-tuning, not chat. "
        "vLLM can serve the same BF16 weights on GPU. Extra memory versus "
        "the Q4 chat image."
    ),
)

_LFM3_VL_2B = SandboxModel(
    id=LFM3_VL_2B_ID,
    display_name="LFM3 VL 2B (CPU)",
    family="LFM3",
    hf_repo="LiquidAI/LFM3-VL-2B",
    hf_gguf_repo="LiquidAI/LFM3-VL-2B-GGUF",
    gguf_filename="LFM3-VL-2B-Q4_K_M.gguf",
    runtime="llama.cpp",
    min_runtime_version=LFM3_MIN_LLAMA_CPP,
    parameters="1.92B",
    weight_bytes=1_099_627_808,
    recommended_vcpu=2,
    recommended_memory_gib=6,
    recommended_disk_gib=12,
    standard_template=LFM3_VL_2B_ID,
    sealed_image=LFM3_VL_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=_license_url("LiquidAI/LFM3-VL-2B"),
    gpu_required=False,
    live=False,
    mmproj_filename="mmproj-LFM3-VL-2B-Q8_0.gguf",
    mmproj_bytes=203_031_872,
    weight_format="gguf",
    runtimes=("llama.cpp", "transformers"),
    input_modalities=("text", "image", "video"),
    output_modalities=("text",),
    sandbox_input_modalities=("text", "image"),
    sandbox_output_modalities=("text",),
    sandbox_notes=(
        "llama.cpp b11938 (PR 31252) accepts images. Video input is not "
        "supported in llama.cpp yet. vLLM support is in progress on GPU."
    ),
)

_LFM3_AUDIO_2B = SandboxModel(
    id=LFM3_AUDIO_2B_ID,
    display_name="LFM3 Audio 2B (CPU)",
    family="LFM3",
    hf_repo="LiquidAI/LFM3-Audio-2B",
    hf_gguf_repo="LiquidAI/LFM3-Audio-2B-GGUF",
    gguf_filename="LFM3-Audio-2B-Q4_K_M.gguf",
    runtime="llama.cpp",
    min_runtime_version=LFM3_AUDIO_MIN_LLAMA_CPP,
    parameters="2B",
    weight_bytes=1_099_626_016,
    recommended_vcpu=2,
    recommended_memory_gib=6,
    recommended_disk_gib=12,
    standard_template=LFM3_AUDIO_2B_ID,
    sealed_image=LFM3_AUDIO_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=_license_url("LiquidAI/LFM3-Audio-2B"),
    gpu_required=False,
    live=False,
    mmproj_filename="mmproj-LFM3-Audio-2B-Q8_0.gguf",
    mmproj_bytes=337_862_080,
    weight_format="gguf",
    runtimes=("llama.cpp", "transformers"),
    input_modalities=("text", "audio"),
    output_modalities=("text", "speech"),
    sandbox_input_modalities=("text", "audio"),
    sandbox_output_modalities=("text", "speech"),
    sandbox_notes=(
        "llama.cpp b11941 (PR 31257). Text and audio in. Text and speech out "
        "via llama-server and llama-tts. Extra memory for the audio projector."
    ),
)

_LFM3_OMNI_2B = SandboxModel(
    id=LFM3_OMNI_2B_ID,
    display_name="LFM3 Omni 2B (CPU)",
    family="LFM3",
    hf_repo="LiquidAI/LFM3-Omni-2B",
    hf_gguf_repo="LiquidAI/LFM3-Omni-2B-GGUF",
    gguf_filename="LFM3-Omni-2B-Q4_K_M.gguf",
    runtime="llama.cpp",
    min_runtime_version=LFM3_AUDIO_MIN_LLAMA_CPP,
    parameters="2.17B",
    weight_bytes=1_099_630_080,
    recommended_vcpu=2,
    recommended_memory_gib=8,
    recommended_disk_gib=14,
    standard_template=LFM3_OMNI_2B_ID,
    sealed_image=LFM3_OMNI_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=_license_url("LiquidAI/LFM3-Omni-2B"),
    gpu_required=False,
    live=False,
    mmproj_filename="mmproj-LFM3-Omni-2B-Q8_0.gguf",
    mmproj_bytes=540_892_768,
    weight_format="gguf",
    runtimes=("llama.cpp", "transformers"),
    input_modalities=("text", "image", "video", "audio"),
    output_modalities=("text", "speech"),
    sandbox_input_modalities=("text", "image", "audio"),
    sandbox_output_modalities=("text", "speech"),
    sandbox_notes=(
        "llama.cpp b11941 (PR 31257). Video input is not supported in "
        "llama.cpp yet. Extra memory for the combined vision and audio "
        "projector (about 516 MiB Q8)."
    ),
)

_D1_2B = SandboxModel(
    id=D1_2B_ID,
    display_name="d1 2B (CPU)",
    family="d1",
    hf_repo="LiquidAI/d1-2B",
    hf_gguf_repo="LiquidAI/d1-2B-GGUF",
    gguf_filename="d1-2B-Q4_K_M.gguf",
    runtime="transformers",
    min_runtime_version=D1_MIN_TRANSFORMERS,
    parameters="1.92B",
    weight_bytes=3_836_417_920,
    recommended_vcpu=2,
    recommended_memory_gib=12,
    recommended_disk_gib=16,
    standard_template=D1_2B_ID,
    sealed_image=D1_2B_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=_license_url("LiquidAI/d1-2B"),
    gpu_required=False,
    live=False,
    mmproj_filename="mmproj-d1-2B-Q8_0.gguf",
    mmproj_bytes=203_031_584,
    weight_format="safetensors",
    runtimes=("transformers",),
    input_modalities=("text", "image", "video"),
    output_modalities=("decision",),
    sandbox_input_modalities=("text", "image"),
    sandbox_output_modalities=("decision",),
    sandbox_notes=(
        "Decision model. One forward pass, no generated tokens. GGUF is "
        "published (Q4_K_M 1,099,629,888 bytes) but llama.cpp support is "
        "pending ggml-org/llama.cpp#31275. Sandbox image uses Transformers "
        "BF16. Video is not wired in the sandbox recipe. GPU is not required "
        "but BF16 on CPU needs about 12 GiB."
    ),
)

_D1_OMNI_2B = SandboxModel(
    id=D1_OMNI_2B_ID,
    display_name="d1 Omni 2B (CPU)",
    family="d1",
    hf_repo="LiquidAI/d1-Omni-2B",
    hf_gguf_repo="LiquidAI/d1-Omni-2B-GGUF",
    gguf_filename="d1-Omni-2B-Q8_0.gguf",
    runtime="transformers",
    min_runtime_version=D1_MIN_TRANSFORMERS,
    parameters="2.17B",
    weight_bytes=4_333_941_640,
    recommended_vcpu=2,
    recommended_memory_gib=14,
    recommended_disk_gib=18,
    standard_template=D1_OMNI_2B_ID,
    sealed_image=D1_OMNI_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=_license_url("LiquidAI/d1-Omni-2B"),
    gpu_required=False,
    live=False,
    mmproj_filename="mmproj-d1-Omni-2B-Q8_0.gguf",
    mmproj_bytes=326_104_096,
    weight_format="safetensors",
    runtimes=("transformers",),
    input_modalities=("text", "image", "video", "audio"),
    output_modalities=("decision",),
    sandbox_input_modalities=("text", "image", "audio"),
    sandbox_output_modalities=("decision",),
    sandbox_notes=(
        "Decision model. No Q4_K_M GGUF is published, only Q8_0 / F16 / BF16. "
        "llama.cpp support is pending ggml-org/llama.cpp#31275. Sandbox image "
        "uses Transformers BF16. Video is not wired in the sandbox recipe. "
        "GPU is not required but BF16 on CPU needs about 14 GiB."
    ),
)

_MODELS: Mapping[str, SandboxModel] = MappingProxyType(
    {
        model.id: model
        for model in (
            _LFM3_1_7B,
            _LFM3_1_7B_BASE,
            _LFM3_VL_2B,
            _LFM3_AUDIO_2B,
            _LFM3_OMNI_2B,
            _D1_2B,
            _D1_OMNI_2B,
        )
    }
)


def list_models() -> tuple[SandboxModel, ...]:
    return tuple(_MODELS[key] for key in sorted(_MODELS))


def get_model(model_id: str) -> SandboxModel:
    try:
        return _MODELS[model_id]
    except KeyError as exc:
        raise UnknownSandboxModel(f"unknown sandbox model: {model_id!r}") from exc


def _template_description(model: SandboxModel) -> str:
    gpu = "GPU is required." if model.gpu_required else "GPU is not required."
    if model.runtime == "llama.cpp" and model.gguf_filename:
        weights = model.gguf_filename
    else:
        weights = f"{model.weight_format} {model.hf_repo}"
    inputs = ", ".join(model.sandbox_input_modalities)
    outputs = ", ".join(model.sandbox_output_modalities)
    return (
        f"{model.family} {model.parameters} on CPU via {model.runtime} "
        f"({weights}). Sandbox inputs: {inputs}. Sandbox outputs: {outputs}. {gpu}"
    )


def catalog_templates() -> tuple[Mapping[str, object], ...]:
    """Entries shaped like ``GET /v1/sandboxes/catalog`` ``templates[]``."""
    templates = []
    for model in list_models():
        templates.append(
            MappingProxyType(
                {
                    "name": model.standard_template,
                    "display_name": model.display_name,
                    "description": _template_description(model),
                    "image": model.sealed_image,
                    "resources": {
                        "cpu": model.recommended_vcpu,
                        "memory_gib": model.recommended_memory_gib,
                    },
                    "disk_gib": model.recommended_disk_gib,
                    "network_modes": ["internet", "deny_all", "allowlist"],
                    "egress_allow": [],
                    "gpu_required": model.gpu_required,
                    "live": model.live,
                    "runtime": model.runtime,
                    "runtimes": list(model.runtimes),
                    "input_modalities": list(model.sandbox_input_modalities),
                    "output_modalities": list(model.sandbox_output_modalities),
                    "license_id": model.license_id,
                }
            )
        )
    return tuple(templates)


def resolve_create(
    model_id: str,
    tier: SandboxTier,
    *,
    name: str | None = None,
    lifetime_seconds: int = 3600,
    max_spend_usd: str = "1.00",
    ssh_public_key: str | None = None,
) -> dict[str, object]:
    """Return the existing-API create body for ``model_id`` on ``tier``."""
    model = get_model(model_id)
    match tier:
        case SandboxTier.STANDARD:
            return {
                "template": model.standard_template,
                "network": "deny_all",
                "lifetime_seconds": lifetime_seconds,
                "max_spend_usd": max_spend_usd,
                "labels": {"cathedral.model": model.id},
            }
        case SandboxTier.SEALED:
            minutes = max(5, (lifetime_seconds + 59) // 60)
            body: dict[str, object] = {
                "profile": "custom.v1",
                "name": name or f"{model.id}-sealed",
                "lifetime": {
                    "mode": "bounded_service",
                    "reuse": "allowed",
                    "max_runtime_minutes": minutes,
                },
                "workload": {"image": model.sealed_image},
                "resources": {
                    "hardware_class": "tdx_cpu",
                    "cpu": 4,
                    "memory_gib": max(16, model.recommended_memory_gib),
                },
                "budget": {
                    "max_spend_usd": float(max_spend_usd),
                    "auto_stop": True,
                },
            }
            if ssh_public_key:
                body["ssh_public_key"] = ssh_public_key
            return body
        case _:
            assert_never(tier)
