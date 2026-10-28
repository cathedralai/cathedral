"""Named sandbox models that resolve to the existing create contracts.

Standard sandboxes use ``POST /v1/sandboxes`` with a catalog ``template``.
Sealed sandboxes use ``POST /v1/workers`` with ``profile: custom.v1`` and
``workload.image``. This package does not invent a new create field.
"""

from .catalog import (
    D1_2B_ID,
    D1_OMNI_2B_ID,
    LFM3_1_7B_BASE_ID,
    LFM3_1_7B_ID,
    LFM3_AUDIO_2B_ID,
    LFM3_OMNI_2B_ID,
    LFM3_VL_2B_ID,
    LFM_OPEN_LICENSE_URL,
    LFM_OPEN_LICENSE_VERSION,
    SandboxModel,
    SandboxTier,
    catalog_templates,
    get_model,
    list_models,
    resolve_create,
)
from .license_terms import (
    COMMERCIAL_REVENUE_THRESHOLD_USD,
    LFM_OPEN_LICENSE_SECTION_5,
    license_blocks_unrestricted_commercial_hosting,
)

__all__ = [
    "COMMERCIAL_REVENUE_THRESHOLD_USD",
    "D1_2B_ID",
    "D1_OMNI_2B_ID",
    "LFM3_1_7B_BASE_ID",
    "LFM3_1_7B_ID",
    "LFM3_AUDIO_2B_ID",
    "LFM3_OMNI_2B_ID",
    "LFM3_VL_2B_ID",
    "LFM_OPEN_LICENSE_SECTION_5",
    "LFM_OPEN_LICENSE_URL",
    "LFM_OPEN_LICENSE_VERSION",
    "SandboxModel",
    "SandboxTier",
    "catalog_templates",
    "get_model",
    "license_blocks_unrestricted_commercial_hosting",
    "list_models",
    "resolve_create",
]
