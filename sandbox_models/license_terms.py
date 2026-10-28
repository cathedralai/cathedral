"""LFM Open License v1.0 facts used by the catalog and the PR description.

Source: https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE
Guide: https://docs.liquid.ai/lfm/help/model-license
"""

from __future__ import annotations

COMMERCIAL_REVENUE_THRESHOLD_USD = 10_000_000

LFM_OPEN_LICENSE_SECTION_5 = (
    "5. Commercial Use Limitation.\n"
    "(a) The rights granted under this License for Commercial Use are "
    "conditioned upon You or Your Legal Entity not exceeding the Threshold.\n"
    "(b) Any Commercial Use of the Work or a Derivative Work by a Legal Entity "
    "that exceeds the Threshold is not licensed under this Agreement.\n"
    "(c) The Threshold shall not apply to a Qualified Non-Profit Organization's "
    "use of the Work or a Derivative Work for Non-Commercial or Research Purposes."
)

LFM_OPEN_LICENSE_THRESHOLD_DEFINITION = (
    '"Threshold" shall mean annual revenue of 10 million United States dollars '
    "($10,000,000) or more."
)

LFM_OPEN_LICENSE_COMMERCIAL_USE_DEFINITION = (
    '"Commercial Use" shall mean any use of the Work for direct or indirect '
    "commercial advantage or monetary compensation."
)


def license_blocks_unrestricted_commercial_hosting() -> bool:
    """True: the open license does not grant commercial use at or above $10M revenue."""
    return True
