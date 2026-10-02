from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_URL = "https://github.com/cathedralai/cathedral-validator"


def test_legacy_package_does_not_publish_validator_commands() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert "cathedral-validator" not in scripts
    assert "cathedral-thin-validator" not in scripts


def test_active_docs_route_validator_operators_to_canonical_repository() -> None:
    docs = (
        ROOT / "README.md",
        ROOT / "VALIDATOR.md",
        ROOT / "docs/PROVENANCE.md",
        ROOT / "docs/THIN_SUBNET_RUNBOOK.md",
        ROOT / "docs/VIOLET_EXTERNAL_SCORES.md",
    )

    for path in docs:
        text = path.read_text("utf-8")
        assert CANONICAL_URL in text, path
        assert "cathedral-validator serve" not in text, path
        assert "cathedral-thin-validator --broadcast" not in text, path
        assert "derived extraction" not in text, path


def test_legacy_config_and_unit_fixtures_are_marked_not_for_install() -> None:
    configs = (
        ROOT / "config/validator.toml",
        ROOT / "config/validator-mainnet-sn39.toml",
        ROOT / "config/validator-mainnet-sn39-launch.toml",
        ROOT / "config/validator-thin-sn39-relay.toml",
    )
    units = (
        ROOT / "deploy/sn39/cathedral-validator-sn39.service",
        ROOT / "deploy/sn39/cathedral-validator-sn39-launch.service",
        ROOT / "deploy/sn39/cathedral-validator-sn39-reconcile.service",
        ROOT / "deploy/sn39/cathedral-sn39-public-status.service",
        ROOT / "deploy/thin/cathedral-thin-validator.service",
    )

    for path in configs:
        text = path.read_text("utf-8")
        assert "LEGACY TEST FIXTURE" in text, path
        assert CANONICAL_URL in text, path

    for path in units:
        text = path.read_text("utf-8")
        assert "Legacy migration fixture" in text, path
        assert CANONICAL_URL in text, path


def test_writer_conflict_guards_remain_intact() -> None:
    units = (
        ROOT / "deploy/sn39/cathedral-validator-sn39.service",
        ROOT / "deploy/sn39/cathedral-validator-sn39-launch.service",
        ROOT / "deploy/sn39/cathedral-validator-sn39-reconcile.service",
    )
    guarded_writers = (
        "cathedral-thin-validator.service",
        "cathedral-confidential-validator-sn39.service",
        "cathedral-confidential-validator.service",
        "cathedral-validator.service",
    )

    for path in units:
        text = path.read_text("utf-8")
        assert "Conflicts=" in text, path
        assert "ExecStartPre=" in text, path
        for writer in guarded_writers:
            assert writer in text, (path, writer)


def test_top_level_v0_launch_docs_are_marked_historical() -> None:
    docs = (
        "AUDIT_ARENA_V0.md",
        "CATHEDRAL_V0_LANES.md",
        "DISTILLATION_READINESS.md",
        "LANE2_SECURE_COMPUTE_PLAN.md",
        "LAUNCH_ANNOUNCEMENT_DRAFT.md",
        "LAUNCH_MERGE_NOTE.md",
        "LAUNCH_READINESS_CHECKLIST.md",
        "LAUNCH_SCORECARD.md",
        "LAUNCH_V0_RUNBOOK.md",
        "SOLVER_ATTESTATION_STATUS.md",
        "TEE_GPU_CAPACITY.md",
    )

    for name in docs:
        head = (ROOT / name).read_text("utf-8").split("\n", 9)[:9]
        assert "> [!WARNING]" in head, name
        assert any(line.startswith("> **Historical record.**") for line in head), name
        assert CANONICAL_URL in "\n".join(head), name


def test_system_map_does_not_present_the_retired_publisher_as_current() -> None:
    text = (ROOT / "README.md").read_text("utf-8")
    system_map = text.split("## How the system fits together", 1)[1].split("## ", 1)[0]
    assert "supported direct-validator source" in system_map
    assert "not an input to the supported direct validator" in system_map
    assert "It derives its own weight vector" in system_map
    assert "https://github.com/cathedralai/cathedral-sandbox" in text
