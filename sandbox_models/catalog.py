"""Catalog of models a caller can pick when creating a Cathedral sandbox.

The live Standard catalog is ``GET /v1/sandboxes/catalog``. The live sealed
catalog is ``GET /v1/profiles`` (``custom.v1``). This module is the source of
truth in this repository for the LFM3 option. The hotel broker in
cathedral-sandbox and the private service layer must consume the same
template name and image before the option is live.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, assert_never


LFM3_1_7B_ID = "lfm3-1.7b"
LFM_OPEN_LICENSE_VERSION = "LFM Open License v1.0"
LFM_OPEN_LICENSE_URL = "https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE"
LFM_OPEN_LICENSE_GUIDE_URL = "https://docs.liquid.ai/lfm/help/model-license"

# Built from images/lfm3-1.7b. Not published to a registry by this change.
LFM3_IMAGE_REF = "cathedral/lfm3-1.7b:q4_k_m"
LFM3_GGUF_REPO = "LiquidAI/LFM3-1.7B-GGUF"
LFM3_GGUF_FILENAME = "LFM3-1.7B-Q4_K_M.gguf"
LFM3_WEIGHT_BYTES = 1_099_600_000
LFM3_MIN_LLAMA_CPP = "b11938"


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
    recommended_vcpu=1,
    recommended_memory_gib=4,
    recommended_disk_gib=10,
    standard_template=LFM3_1_7B_ID,
    sealed_image=LFM3_IMAGE_REF,
    license_id=LFM_OPEN_LICENSE_VERSION,
    license_url=LFM_OPEN_LICENSE_URL,
    gpu_required=False,
    live=False,
)

_MODELS: Mapping[str, SandboxModel] = MappingProxyType({_LFM3_1_7B.id: _LFM3_1_7B})


def list_models() -> tuple[SandboxModel, ...]:
    return tuple(_MODELS[key] for key in sorted(_MODELS))


def get_model(model_id: str) -> SandboxModel:
    try:
        return _MODELS[model_id]
    except KeyError as exc:
        raise UnknownSandboxModel(f"unknown sandbox model: {model_id!r}") from exc


def catalog_templates() -> tuple[Mapping[str, object], ...]:
    """Entries shaped like ``GET /v1/sandboxes/catalog`` ``templates[]``."""
    templates = []
    for model in list_models():
        templates.append(
            MappingProxyType(
                {
                    "name": model.standard_template,
                    "display_name": model.display_name,
                    "description": (
                        f"{model.family} {model.parameters} on CPU via {model.runtime} "
                        f"({model.gguf_filename}). GPU is not required."
                    ),
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
                    "memory_gib": 16,
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
