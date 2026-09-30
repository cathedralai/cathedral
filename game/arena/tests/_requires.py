"""Markers for arena tests that need what a plain checkout does not have.

The real audit corpus and the z3 factory live in the sibling audit-hunter checkout,
the pinned AuditTarget harnesses in audit-lane, and minted proofs need z3 as well as
the factory. Without them the arena runs on its bundled fallback, and these tests
skip instead of failing.
"""

import pytest

from game.arena import corpus, mint

# replay.py loads audit-lane from the same parent directory as audit-hunter.
_AUDIT_LANE = corpus._AUDIT_HUNTER.parent / "audit-lane" / "audit_lane.py"

requires_audit_hunter = pytest.mark.skipif(
    not corpus._AUDIT_HUNTER.is_dir(), reason=f"needs the audit-hunter checkout at {corpus._AUDIT_HUNTER}")
requires_audit_lane = pytest.mark.skipif(
    not _AUDIT_LANE.is_file(), reason=f"needs the audit-lane checkout at {_AUDIT_LANE.parent}")
requires_mint = pytest.mark.skipif(
    not mint.can_mint(), reason=f"needs z3-solver and the audit-hunter factory at {mint._FACTORY}")
# The base replay targets hold two exploits; audit-lane and minting each add more.
requires_audit_lane_or_mint = pytest.mark.skipif(
    not (_AUDIT_LANE.is_file() or mint.can_mint()),
    reason="needs audit-lane, or z3-solver and the audit-hunter factory, for a third exploit target")
