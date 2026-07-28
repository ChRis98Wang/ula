#!/usr/bin/env python3
"""Bind a selected Qwen winner and explicit approval into a V8.1 config.

This command never starts training.  It validates the selected winner receipt,
builds a hash-bound approval receipt, validates the derived approved config,
and publishes the config last as the transaction commit point.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import train_hanyang_beat2_emotion_preserving_v81 as trainer  # noqa: E402


PROMOTION_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_promotion_receipt_v8_1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _explicit_approval_identity(
    *, approved_by: str, approved_utc: str, decision_notes: str
) -> tuple[str, str, str]:
    reviewer = str(approved_by).strip()
    utc_text = str(approved_utc).strip()
    notes = str(decision_notes).strip()
    if not reviewer or not utc_text or not notes:
        raise ValueError(
            "approved_by, approved_utc, and decision_notes must be explicit"
        )
    try:
        timestamp = datetime.fromisoformat(
            utc_text.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("approved_utc must be ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ValueError("approved_utc must include an explicit UTC offset")
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("approved_utc must be UTC, not a local timezone")
    canonical_utc = timestamp.isoformat().replace("+00:00", "Z")
    return reviewer, canonical_utc, notes


def promote(
    *,
    base_config_path: str | Path,
    selected_winner_receipt_path: str | Path,
    output_config_path: str | Path,
    output_approval_receipt_path: str | Path,
    approved_by: str,
    approved_utc: str,
    decision_notes: str,
) -> dict[str, Any]:
    """Atomically publish an approved derived config, without training."""

    reviewer, canonical_utc, notes = _explicit_approval_identity(
        approved_by=approved_by,
        approved_utc=approved_utc,
        decision_notes=decision_notes,
    )
    base_path = Path(base_config_path).expanduser().resolve()
    winner_path = Path(selected_winner_receipt_path).expanduser().resolve()
    config_path = Path(output_config_path).expanduser().resolve()
    approval_path = Path(
        output_approval_receipt_path
    ).expanduser().resolve()
    if not base_path.is_file() or not winner_path.is_file():
        raise FileNotFoundError("base config or selected winner receipt is missing")
    if config_path == approval_path or config_path == base_path:
        raise ValueError("promotion output paths must be new and distinct")
    if config_path.exists() or approval_path.exists():
        raise FileExistsError("promotion outputs already exist")

    base_raw = _read_json(base_path)
    base_validated = trainer.validate_config(base_raw)
    if (
        base_validated["qwen_ab_selection_gate"]["winner_selected"] is not False
        or base_validated["approval_gate"]["status"] != "blocked"
    ):
        raise ValueError("promotion base must be the checked blocked config")

    derived = deepcopy(base_raw)
    derived["qwen_ab_selection_gate"] = {
        "required": True,
        "status": "selected_receipt_bound",
        "selection_receipt": str(winner_path),
        "expected_selection_receipt_file_sha256": trainer.sha256_file(
            winner_path
        ),
    }
    selected = trainer._validate_qwen_ab_selection(
        derived["qwen_ab_selection_gate"]
    )
    arms = derived["winner_overlay_arms"]
    arms["shared_selected_foundation_sha256"] = selected[
        "selected_foundation_sha256"
    ]
    arms["shared_selected_qwen_variant"] = selected[
        "selected_qwen_variant"
    ]
    arms["shared_selected_condition_cache_sha256"] = selected[
        "selected_condition_cache_sha256"
    ]
    derived["approval_gate"] = {
        "required": True,
        "status": "blocked",
        "approval_receipt": None,
        "expected_approval_receipt_sha256": None,
    }
    blocked_validated = trainer.validate_config(derived)
    binding_sha256 = trainer.config_without_approval_sha256(
        blocked_validated
    )
    approval = {
        "schema_version": trainer.SCHEMA_VERSION,
        "artifact_kind": trainer.APPROVAL_ARTIFACT_KIND,
        "decision": "approved",
        "training_launch_allowed": True,
        "approved_by": reviewer,
        "approved_utc": canonical_utc,
        "decision_notes": notes,
        "config_without_approval_sha256": binding_sha256,
        "foundation_checkpoint_sha256": selected[
            "selected_foundation_sha256"
        ],
        "winner_selection_receipt_file_sha256": trainer.sha256_file(
            winner_path
        ),
        "winner_selection_receipt_canonical_sha256": selected[
            "selection_receipt_canonical_sha256"
        ],
        "winner_invariant_contract_sha256": selected[
            "invariant_contract_sha256"
        ],
        "selected_qwen_variant": selected["selected_qwen_variant"],
        "formal_release_eligible": False,
    }
    approval.update(trainer._hanyang_artifact_audit(blocked_validated))
    approval["sha256"] = trainer.canonical_sha256(approval)
    _atomic_json(approval, approval_path)
    approval_file_sha256 = trainer.sha256_file(approval_path)
    derived["approval_gate"] = {
        "required": True,
        "status": "approved",
        "approval_receipt": str(approval_path),
        "expected_approval_receipt_sha256": approval_file_sha256,
    }
    try:
        validated = trainer.validate_config(derived)
        plans = {
            arm: trainer.build_lineage_contract(validated, arm=arm)
            for arm in (
                "winner_control_0pct_hanyang",
                "winner_isolated_5pct_hanyang",
            )
        }
        _atomic_json(derived, config_path)
        round_trip = trainer.read_config(config_path)
        if round_trip["approval_gate"]["status"] != "approved":
            raise RuntimeError("published config lost approval")
    except Exception:
        if approval_path.is_file() and not config_path.exists():
            approval_path.unlink()
        raise

    runner = PROJECT_ROOT / "tools/train_hanyang_beat2_emotion_preserving_v81.py"
    smoke_commands = {
        arm: " ".join(
            shlex.quote(value)
            for value in (
                sys.executable,
                str(runner),
                "--config",
                str(config_path),
                "--arm",
                arm,
                "--smoke-test",
                "--smoke-output-dir",
                str(config_path.parent / f"{arm}_smoke"),
                "--device",
                "cpu",
            )
        )
        for arm in (
            "winner_control_0pct_hanyang",
            "winner_isolated_5pct_hanyang",
        )
    }
    result = {
        "schema_version": trainer.SCHEMA_VERSION,
        "artifact_kind": PROMOTION_ARTIFACT_KIND,
        "training_started": False,
        "derived_config": str(config_path),
        "derived_config_sha256": trainer.sha256_file(config_path),
        "approval_receipt": str(approval_path),
        "approval_receipt_sha256": approval_file_sha256,
        "winner_selection_receipt": str(winner_path),
        "winner_selection_receipt_sha256": trainer.sha256_file(winner_path),
        "selected_qwen_variant": selected["selected_qwen_variant"],
        "selected_foundation_sha256": selected[
            "selected_foundation_sha256"
        ],
        "preflight_plans": plans,
        "cpu_smoke_commands": smoke_commands,
        "formal_systemd_launch_allowed_by_this_command": False,
    }
    result.update(trainer._hanyang_artifact_audit(validated))
    result["sha256"] = trainer.canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a selected winner into an approved V8.1 config"
    )
    parser.add_argument(
        "--base-config",
        default=str(
            PROJECT_ROOT
            / "configs/hanyang_beat2_emotion_preserving_v81.json"
        ),
    )
    parser.add_argument("--selected-winner-receipt", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-approval-receipt", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-utc", required=True)
    parser.add_argument("--decision-notes", required=True)
    args = parser.parse_args()
    result = promote(
        base_config_path=args.base_config,
        selected_winner_receipt_path=args.selected_winner_receipt,
        output_config_path=args.output_config,
        output_approval_receipt_path=args.output_approval_receipt,
        approved_by=args.approved_by,
        approved_utc=args.approved_utc,
        decision_notes=args.decision_notes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
