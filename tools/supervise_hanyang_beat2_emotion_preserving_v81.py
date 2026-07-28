#!/usr/bin/env python3
"""Fail-closed, restartable night supervisor for the approved V8.1 pair.

This process only observes the already-running Qwen/video systemd units.  It
never starts, stops, or kills a unit or GPU process.  Transient units may be
garbage-collected after success, so their sealed, hash-validated artifacts are
the durable completion evidence.  Once all prerequisites are sealed, it
promotes the checked blocked config, runs two isolated CUDA smokes, and then
runs control followed by treatment.  Each arm initializes independently from
the same selected winner; control is never a warm start for treatment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import promote_hanyang_beat2_emotion_preserving_v81 as promoter  # noqa: E402
from tools import select_beat2_emotion_hierarchy_v7_qwen_winner as selector  # noqa: E402
from tools import train_hanyang_beat2_emotion_preserving_v81 as trainer  # noqa: E402


SCHEMA_VERSION = "8.1"
CONFIG_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_night_supervisor_config_v8_1"
)
STATE_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_night_supervisor_state_v8_1"
)
RECEIPT_ARTIFACT_KIND = (
    "hanyang_beat2_emotion_preserving_night_supervisor_receipt_v8_1"
)
VIDEO_SUMMARY_ARTIFACT_KIND = (
    "beat2_v7_qwen_ab_gt_60s_comparison_v1"
)
VIDEO_FILENAME = "BEAT2_GT_vs_frozen_Qwen_vs_LoRA_Qwen_60s.mp4"
ARMS = (
    "winner_control_0pct_hanyang",
    "winner_isolated_5pct_hanyang",
)
STAGES = (
    "prerequisites",
    "promotion",
    "control_smoke",
    "treatment_smoke",
    "control_formal",
    "treatment_formal",
)
STAGE_ARM = {
    "control_smoke": ARMS[0],
    "treatment_smoke": ARMS[1],
    "control_formal": ARMS[0],
    "treatment_formal": ARMS[1],
}


class SupervisorError(RuntimeError):
    """Terminal fail-closed supervisor error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("sha256", None)
    return hashlib.sha256(_stable_json(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path, *, context: str) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"{context} is unreadable: {target}") from exc
    if not isinstance(value, dict):
        raise SupervisorError(f"{context} must be a JSON object")
    return value


def _atomic_json(value: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, target)


def _absolute_path(value: object, *, field: str) -> str:
    text = str(value or "")
    path = Path(text).expanduser()
    if not text or not path.is_absolute():
        raise SupervisorError(f"{field} must be an explicit absolute path")
    return str(path.resolve())


def read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = _read_json(config_path, context="supervisor config")
    exact = {
        "schema_version",
        "artifact_kind",
        "base_blocked_config",
        "selected_winner_receipt",
        "qwen_b_training_service_unit",
        "video_waiter_service_unit",
        "video_summary",
        "video_mp4",
        "promoted_config",
        "approval_receipt",
        "control_smoke_output_dir",
        "treatment_smoke_output_dir",
        "state",
        "receipt",
        "python_executable",
        "trainer",
        "promoter",
        "gpu",
        "data_contract",
    }
    if set(raw) != exact:
        raise SupervisorError("supervisor config fields changed")
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("artifact_kind") != CONFIG_ARTIFACT_KIND
        or raw.get("qwen_b_training_service_unit")
        != "beat2-emotion-hierarchy-v7-lora-qwen.service"
        or raw.get("video_waiter_service_unit")
        != "beat2-qwen-ab-gt-60s-after-train.service"
    ):
        raise SupervisorError("supervisor identity or prerequisite units changed")
    paths = (
        "base_blocked_config",
        "selected_winner_receipt",
        "video_summary",
        "video_mp4",
        "promoted_config",
        "approval_receipt",
        "control_smoke_output_dir",
        "treatment_smoke_output_dir",
        "state",
        "receipt",
        "python_executable",
        "trainer",
        "promoter",
    )
    values = deepcopy(raw)
    for field in paths:
        values[field] = _absolute_path(values[field], field=field)
    if Path(values["video_mp4"]).name != VIDEO_FILENAME:
        raise SupervisorError("final comparison MP4 path changed")
    if Path(values["state"]) == Path(values["receipt"]):
        raise SupervisorError("state and receipt paths must be distinct")
    if Path(values["control_smoke_output_dir"]) == Path(
        values["treatment_smoke_output_dir"]
    ):
        raise SupervisorError("smoke output directories must be unique")
    gpu = values.get("gpu")
    if (
        not isinstance(gpu, Mapping)
        or set(gpu)
        != {"index", "minimum_free_memory_mib", "unknown_compute_policy"}
        or type(gpu.get("index")) is not int
        or int(gpu["index"]) < 0
        or not math.isfinite(float(gpu.get("minimum_free_memory_mib", 0)))
        or float(gpu["minimum_free_memory_mib"]) < 24576
        or gpu.get("unknown_compute_policy") != "record_never_kill"
    ):
        raise SupervisorError("GPU safety gate changed or is below 24 GiB")
    data = values.get("data_contract")
    if (
        not isinstance(data, Mapping)
        or set(data)
        != {
            "hanyang_strict_count",
            "hanyang_boundary_admitted_count",
            "kimodo_admitted_count",
            "formal_arms_sequential",
            "arm_initialization",
        }
        or data.get("hanyang_strict_count") != 344
        or data.get("hanyang_boundary_admitted_count") != 0
        or data.get("kimodo_admitted_count") != 0
        or data.get("formal_arms_sequential") is not True
        or data.get("arm_initialization")
        != "same_selected_winner_independent_no_cross_arm_warm_start"
    ):
        raise SupervisorError("supervisor data/sequential contract changed")
    for field in ("base_blocked_config", "python_executable", "trainer", "promoter"):
        if not Path(values[field]).is_file():
            raise SupervisorError(f"required file is missing: {field}")
    values["_config_path"] = str(config_path)
    values["_config_sha256"] = sha256_file(config_path)
    return values


def _validate_identity(
    approved_by: str, approved_utc: str, decision_notes: str
) -> dict[str, str]:
    reviewer, canonical_utc, notes = promoter._explicit_approval_identity(
        approved_by=approved_by,
        approved_utc=approved_utc,
        decision_notes=decision_notes,
    )
    return {
        "approved_by": reviewer,
        "approved_utc": canonical_utc,
        "decision_notes": notes,
    }


def _new_state(
    config: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    state = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": STATE_ARTIFACT_KIND,
        "status": "running",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "config": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "approval_identity": dict(identity),
        "current_stage": "prerequisites",
        "stages": {},
        "wait_observations": [],
        "failure": None,
    }
    state["sha256"] = canonical_sha256(state)
    return state


def _load_or_create_state(
    config: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    path = Path(config["state"])
    if not path.is_file():
        state = _new_state(config, identity)
        _atomic_json(state, path)
        return state
    state = _read_json(path, context="supervisor state")
    if (
        state.get("artifact_kind") != STATE_ARTIFACT_KIND
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("sha256") != canonical_sha256(state)
        or state.get("config_sha256") != config["_config_sha256"]
        or state.get("approval_identity") != dict(identity)
        or not isinstance(state.get("stages"), Mapping)
        or not isinstance(state.get("wait_observations"), list)
    ):
        raise SupervisorError("existing supervisor state binding is invalid")
    if state.get("status") == "failed":
        raise SupervisorError("supervisor state is terminally failed; no skipping")
    return state


def _save_state(config: Mapping[str, Any], state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    state["sha256"] = canonical_sha256(state)
    _atomic_json(state, config["state"])


def _unit_snapshot(
    unit: str,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
    allow_transient_reclaimed: bool = False,
) -> dict[str, Any]:
    argv = [
        "systemctl",
        "--user",
        "show",
        unit,
        "--property=LoadState",
        "--property=ActiveState",
        "--property=SubState",
        "--property=Result",
        "--property=ExecMainStatus",
        "--no-pager",
    ]
    result = command_runner(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SupervisorError(f"cannot inspect required systemd unit: {unit}")
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    required = {"LoadState", "ActiveState", "SubState", "Result", "ExecMainStatus"}
    if set(fields) != required:
        raise SupervisorError(f"required systemd unit is unavailable: {unit}")
    if fields["LoadState"] != "loaded":
        reclaimed = (
            allow_transient_reclaimed
            and fields["LoadState"] == "not-found"
            and fields["ActiveState"] == "inactive"
            and fields["SubState"] == "dead"
        )
        if not reclaimed:
            raise SupervisorError(f"required systemd unit is unavailable: {unit}")
        return {
            "unit": unit,
            "argv": argv,
            **fields,
            "available": False,
            "transient_unit_reclaimed": True,
            "running": False,
            "succeeded_and_exited": False,
            "failed": False,
        }
    active = fields["ActiveState"] in {"active", "activating", "reloading"}
    success = (
        not active
        and fields["ActiveState"] == "inactive"
        and fields["Result"] == "success"
        and fields["ExecMainStatus"] == "0"
    )
    failed = fields["ActiveState"] == "failed" or (
        not active and not success
    )
    return {
        "unit": unit,
        "argv": argv,
        **fields,
        "available": True,
        "transient_unit_reclaimed": False,
        "running": active,
        "succeeded_and_exited": success,
        "failed": failed,
    }


def _selected_receipt_snapshot(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {"ready": False, "reason": "selected_receipt_missing"}
    receipt = _read_json(target, context="winner selection receipt")
    try:
        checked = selector.validate_selection_receipt(
            receipt, require_selected=False
        )
    except ValueError as exc:
        raise SupervisorError("winner selection receipt failed validation") from exc
    if checked.get("status") == selector.PENDING_STATUS:
        return {
            "ready": False,
            "reason": "winner_selection_pending",
            "file_sha256": sha256_file(target),
            "receipt_sha256": checked["sha256"],
        }
    file_sha = sha256_file(target)
    try:
        bound = trainer._validate_qwen_ab_selection(
            {
                "required": True,
                "status": "selected_receipt_bound",
                "selection_receipt": str(target.resolve()),
                "expected_selection_receipt_file_sha256": file_sha,
            }
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SupervisorError("selected winner is not bindable to V8.1") from exc
    return {
        "ready": True,
        "file_sha256": file_sha,
        "receipt_sha256": checked["sha256"],
        "selected_qwen_variant": bound["selected_qwen_variant"],
        "selected_foundation_sha256": bound["selected_foundation_sha256"],
        "selected_condition_cache_sha256": bound[
            "selected_condition_cache_sha256"
        ],
        "invariant_contract_sha256": bound["invariant_contract_sha256"],
    }


def _validate_archived_qwen_b_completion(
    selected_receipt_path: str | Path,
) -> dict[str, Any]:
    """Revalidate durable B artifacts when its transient unit is gone.

    The selected receipt is necessary but not sufficient: both referenced B
    files are hashed again and the selector's full arm validator is rerun.
    This prevents a stale or tampered artifact from standing in for service
    success.
    """

    receipt_path = Path(selected_receipt_path)
    if not receipt_path.is_file():
        raise SupervisorError(
            "cannot prove Qwen B completion without selected receipt"
        )
    receipt = _read_json(
        receipt_path, context="winner selection receipt for Qwen B archive"
    )
    try:
        checked = selector.validate_selection_receipt(
            receipt, require_selected=True
        )
        arm = (checked.get("arms") or {}).get(selector.LORA_VARIANT)
        if not isinstance(arm, Mapping) or arm.get("completed") is not True:
            raise ValueError("selected receipt has no completed LoRA B arm")
        summary_record = arm.get("summary")
        checkpoint_record = arm.get("checkpoint")
        if not isinstance(summary_record, Mapping) or not isinstance(
            checkpoint_record, Mapping
        ):
            raise ValueError("selected receipt has incomplete LoRA B artifacts")
        summary_path = Path(str(summary_record.get("path", "")))
        checkpoint_path = Path(str(checkpoint_record.get("path", "")))
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("sealed LoRA B artifact is missing")
        summary_sha = sha256_file(summary_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        if (
            summary_sha != summary_record.get("file_sha256")
            or checkpoint_sha != checkpoint_record.get("sha256")
            or int(summary_record.get("completed_steps", -1))
            != selector.EXPECTED_STEPS
            or summary_record.get("status") != "experimental_candidate"
            or arm.get("no_external_data") is not True
            or arm.get("no_kimodo") is not True
            or arm.get("no_hanyang") is not True
            or (arm.get("anti_collapse_gate") or {}).get("passed") is not True
        ):
            raise ValueError("sealed LoRA B hashes or completion contract changed")
        sealed_record = checked.get("sealed_ab_receipt")
        if not isinstance(sealed_record, Mapping):
            raise ValueError("selected receipt lost sealed A/B binding")
        sealed = selector.validate_sealed_ab_receipt(
            sealed_record["path"],
            expected_file_sha256=sealed_record["file_sha256"],
        )
        revalidated = selector._validate_arm(
            variant=selector.LORA_VARIANT,
            summary_path=summary_path,
            checkpoint_path=checkpoint_path,
            sealed=sealed,
            expected_summary_sha256=summary_record["file_sha256"],
            expected_checkpoint_sha256=checkpoint_record["sha256"],
        )
        if revalidated != dict(arm):
            raise ValueError("revalidated LoRA B differs from selected receipt")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise SupervisorError(
            "archived Qwen B completion evidence failed validation"
        ) from exc
    return {
        "status": "complete",
        "evidence": "selected_receipt_plus_full_lora_b_artifact_revalidation",
        "selected_receipt": str(receipt_path.resolve()),
        "selected_receipt_file_sha256": sha256_file(receipt_path),
        "selected_receipt_canonical_sha256": checked["sha256"],
        "summary": str(summary_path.resolve()),
        "summary_sha256": summary_sha,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "completed_steps": selector.EXPECTED_STEPS,
    }


def _qwen_completion_snapshot(
    unit_snapshot: Mapping[str, Any],
    *,
    selected_receipt_path: str | Path,
) -> dict[str, Any]:
    if unit_snapshot.get("failed") is True:
        raise SupervisorError("Qwen B training unit failed")
    available = unit_snapshot.get("available", True)
    if available and unit_snapshot.get("succeeded_and_exited") is not True:
        return {
            "ready": False,
            "unit": deepcopy(dict(unit_snapshot)),
            "reason": "qwen_b_training_unit_not_exited",
        }
    archive = _validate_archived_qwen_b_completion(selected_receipt_path)
    return {
        "ready": True,
        "unit": deepcopy(dict(unit_snapshot)),
        "completion_evidence": (
            "loaded_unit_success_plus_sealed_artifacts"
            if available
            else "transient_unit_reclaimed_plus_sealed_artifacts"
        ),
        "archive": archive,
    }


def _validate_video(config: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = Path(config["video_summary"])
    video_path = Path(config["video_mp4"])
    if not summary_path.is_file() or not video_path.is_file():
        raise SupervisorError("video waiter exited without final summary and MP4")
    summary = _read_json(summary_path, context="GT/A/B video summary")
    unsigned = deepcopy(summary)
    expected_internal = unsigned.pop("sha256", None)
    if (
        summary.get("schema_version") != 1
        or summary.get("artifact_kind") != VIDEO_SUMMARY_ARTIFACT_KIND
        or summary.get("status") != "complete"
        or summary.get("no_external_data") is not True
        or summary.get("no_kimodo") is not True
        or summary.get("no_hanyang") is not True
        or expected_internal
        != hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise SupervisorError("GT/A/B video summary contract is invalid")
    final = (summary.get("artifacts") or {}).get("final_video")
    if (
        not isinstance(final, Mapping)
        or Path(str(final.get("path", ""))).resolve() != video_path.resolve()
        or final.get("sha256") != sha256_file(video_path)
        or float(final.get("duration_sec", 0.0)) < 59.0
    ):
        raise SupervisorError("final GT/A/B MP4 receipt is invalid")
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_internal_sha256": expected_internal,
        "video": str(video_path),
        "video_sha256": final["sha256"],
        "duration_sec": float(final["duration_sec"]),
    }


def _video_completion_snapshot(
    config: Mapping[str, Any],
    unit_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if unit_snapshot.get("failed") is True:
        raise SupervisorError("GT/A/B video waiter unit failed")
    available = unit_snapshot.get("available", True)
    if available and unit_snapshot.get("succeeded_and_exited") is not True:
        return {
            "ready": False,
            "unit": deepcopy(dict(unit_snapshot)),
            "reason": "video_waiter_unit_not_exited",
        }
    video = _validate_video(config)
    return {
        "ready": True,
        "unit": deepcopy(dict(unit_snapshot)),
        "completion_evidence": (
            "loaded_unit_success_plus_validated_video_artifacts"
            if available
            else "transient_unit_reclaimed_plus_validated_video_artifacts"
        ),
        "video": video,
    }


def _gpu_snapshot(
    config: Mapping[str, Any],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    gpu = config["gpu"]
    index = str(gpu["index"])
    memory_argv = [
        "nvidia-smi",
        "--id=" + index,
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    memory = command_runner(
        memory_argv, capture_output=True, text=True, check=False
    )
    if memory.returncode != 0:
        raise SupervisorError("nvidia-smi memory query failed")
    try:
        free_mib = float(memory.stdout.strip().splitlines()[0])
    except (IndexError, ValueError) as exc:
        raise SupervisorError("nvidia-smi returned invalid free memory") from exc
    process_argv = [
        "nvidia-smi",
        "--id=" + index,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    processes = command_runner(
        process_argv, capture_output=True, text=True, check=False
    )
    if processes.returncode != 0:
        raise SupervisorError("nvidia-smi compute-process query failed")
    compute: list[dict[str, Any]] = []
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise SupervisorError("nvidia-smi returned invalid compute process")
        try:
            compute.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "used_memory_mib": float(parts[2]),
                }
            )
        except ValueError as exc:
            raise SupervisorError(
                "nvidia-smi returned invalid compute process values"
            ) from exc
    threshold = float(gpu["minimum_free_memory_mib"])
    return {
        "gpu_index": int(gpu["index"]),
        "free_memory_mib": free_mib,
        "minimum_free_memory_mib": threshold,
        "compute_processes": compute,
        "unknown_compute_pids": [item["pid"] for item in compute],
        "memory_threshold_passed": free_mib >= threshold,
        "ready": free_mib >= threshold,
        "policy": "unknown_compute_record_only_never_kill",
        "memory_argv": memory_argv,
        "process_argv": process_argv,
    }


def _validate_promotion_outputs(
    config: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    promoted_path = Path(config["promoted_config"])
    approval_path = Path(config["approval_receipt"])
    if not promoted_path.is_file() or not approval_path.is_file():
        raise SupervisorError("promotion did not publish both bound outputs")
    values = trainer.read_config(promoted_path)
    approval = _read_json(approval_path, context="V8.1 approval receipt")
    selected_sha = sha256_file(config["selected_winner_receipt"])
    if (
        values["approval_gate"]["status"] != "approved"
        or Path(values["approval_gate"]["approval_receipt"]).resolve()
        != approval_path.resolve()
        or values["qwen_ab_selection_gate"]["winner_selected"] is not True
        or Path(values["qwen_ab_selection_gate"]["selection_receipt"]).resolve()
        != Path(config["selected_winner_receipt"]).resolve()
        or approval.get("approved_by") != identity["approved_by"]
        or approval.get("approved_utc") != identity["approved_utc"]
        or approval.get("decision_notes") != identity["decision_notes"]
        or approval.get("winner_selection_receipt_file_sha256") != selected_sha
        or approval.get("training_launch_allowed") is not True
        or approval.get("formal_release_eligible") is not False
    ):
        raise SupervisorError("published promotion binding is invalid")
    return {
        "promoted_config": str(promoted_path),
        "promoted_config_sha256": sha256_file(promoted_path),
        "approval_receipt": str(approval_path),
        "approval_receipt_sha256": sha256_file(approval_path),
        "selected_winner_receipt_sha256": selected_sha,
        "selected_qwen_variant": values["qwen_ab_selection_gate"][
            "selected_qwen_variant"
        ],
    }


def _run_promotion(
    config: Mapping[str, Any],
    identity: Mapping[str, str],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    if Path(config["promoted_config"]).exists() or Path(
        config["approval_receipt"]
    ).exists():
        return _validate_promotion_outputs(config, identity)
    argv = [
        config["python_executable"],
        config["promoter"],
        "--base-config",
        config["base_blocked_config"],
        "--selected-winner-receipt",
        config["selected_winner_receipt"],
        "--output-config",
        config["promoted_config"],
        "--output-approval-receipt",
        config["approval_receipt"],
        "--approved-by",
        identity["approved_by"],
        "--approved-utc",
        identity["approved_utc"],
        "--decision-notes",
        identity["decision_notes"],
    ]
    result = command_runner(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SupervisorError(
            "promotion subprocess failed: " + result.stderr.strip()[-1000:]
        )
    record = _validate_promotion_outputs(config, identity)
    record["argv"] = argv
    return record


def _summary_path(
    config: Mapping[str, Any], *, arm: str, smoke: bool
) -> Path:
    if smoke:
        root = Path(
            config[
                "control_smoke_output_dir"
                if arm == ARMS[0]
                else "treatment_smoke_output_dir"
            ]
        )
    else:
        approved = trainer.read_config(config["promoted_config"])
        root = Path(approved["winner_overlay_arms"][arm]["output_dir"])
    return root / "training_summary_v8_1.json"


def _validate_emotion_candidate_gate(gate: object) -> dict[str, Any]:
    """Independently enforce every emotion-preservation hard gate."""

    names = {
        "aligned_vs_zero",
        "aligned_vs_cross_group",
        "flow_gap",
        "q2_recall",
        "q6_recall",
        "global54_recall",
    }
    if not isinstance(gate, Mapping):
        raise SupervisorError("formal candidate has no diagnostic gate")
    absolute = gate.get("absolute_v7_gate")
    retentions = gate.get("retentions")
    thresholds = gate.get("retention_thresholds")
    checks = gate.get("retention_checks")
    if (
        gate.get("admissible") is not True
        or not isinstance(absolute, Mapping)
        or absolute.get("passed") is not True
        or not isinstance(retentions, Mapping)
        or set(retentions) != names
        or not isinstance(thresholds, Mapping)
        or set(thresholds) != names
        or not isinstance(checks, Mapping)
        or set(checks) != names
        or gate.get("hanyang_validation_finite") is not True
        or gate.get("failure_reasons") != []
    ):
        raise SupervisorError("formal candidate emotion gate is not admissible")
    for name in sorted(names):
        try:
            value = float(retentions[name])
            threshold = float(thresholds[name])
        except (TypeError, ValueError) as exc:
            raise SupervisorError(
                f"formal candidate {name} retention is invalid"
            ) from exc
        if (
            not math.isfinite(value)
            or not math.isfinite(threshold)
            or threshold < 0.9
            or value < threshold
            or checks[name] is not True
        ):
            raise SupervisorError(
                f"formal candidate emotion retention failed: {name}"
            )
    if not math.isclose(
        float(gate.get("minimum_emotion_retention", float("-inf"))),
        min(float(value) for value in retentions.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SupervisorError("formal candidate minimum retention is inconsistent")
    return {
        "absolute_v7_gate_passed": True,
        "retentions": dict(retentions),
        "retention_thresholds": dict(thresholds),
        "retention_checks": dict(checks),
        "hanyang_validation_finite": True,
    }


def _validate_arm_summary(
    config: Mapping[str, Any], *, arm: str, smoke: bool
) -> dict[str, Any]:
    summary_path = _summary_path(config, arm=arm, smoke=smoke)
    if not summary_path.is_file():
        raise SupervisorError(f"{arm} summary is missing")
    summary = _read_json(summary_path, context=f"{arm} training summary")
    expected_steps = 1 if smoke else 60000
    expected_status = (
        "technical_smoke_completed_not_candidate"
        if smoke
        else "candidate_available"
    )
    last = Path(str(summary.get("last_checkpoint", "")))
    state = Path(str(summary.get("state", "")))
    if (
        summary.get("artifact_kind") != trainer.SUMMARY_ARTIFACT_KIND
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("arm") != arm
        or summary.get("completed_steps") != expected_steps
        or summary.get("smoke_test") is not smoke
        or summary.get("run_status") != expected_status
        or summary.get("formal_release_eligible") is not False
        or summary.get("prefix_schedule_assertion_passed") is not True
        or not last.is_file()
        or summary.get("last_checkpoint_sha256") != sha256_file(last)
        or not state.is_file()
        or summary.get("state_sha256") != sha256_file(state)
        or summary.get("exposure") != summary.get("expected_exposure")
    ):
        raise SupervisorError(f"{arm} summary contract failed")
    if smoke:
        if (
            summary.get("candidate_available") is not False
            or summary.get("best_admissible") is not None
            or Path(summary_path.parent / "best_admissible_generator_v8_1.pt").exists()
        ):
            raise SupervisorError("technical smoke was incorrectly candidate-eligible")
        best_path = None
        best_sha = None
    else:
        best = summary.get("best_admissible")
        best_path = Path(str((best or {}).get("checkpoint", "")))
        emotion_gate = _validate_emotion_candidate_gate(
            (best or {}).get("gate")
        )
        if (
            summary.get("candidate_available") is not True
            or not isinstance(best, Mapping)
            or (best.get("gate") or {}).get("admissible") is not True
            or not best_path.is_file()
            or best.get("checkpoint_sha256") != sha256_file(best_path)
        ):
            raise SupervisorError(f"{arm} has no admissible diagnostic candidate")
        best_sha = best["checkpoint_sha256"]
    return {
        "arm": arm,
        "smoke_test": smoke,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "completed_steps": expected_steps,
        "run_status": expected_status,
        "last_checkpoint": str(last),
        "last_checkpoint_sha256": summary["last_checkpoint_sha256"],
        "best_admissible_checkpoint": str(best_path) if best_path else None,
        "best_admissible_checkpoint_sha256": best_sha,
        "emotion_candidate_gate": emotion_gate if not smoke else None,
        "exposure": summary["exposure"],
    }


def _run_arm(
    config: Mapping[str, Any],
    *,
    arm: str,
    smoke: bool,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    summary_path = _summary_path(config, arm=arm, smoke=smoke)
    if summary_path.is_file():
        return _validate_arm_summary(config, arm=arm, smoke=smoke)
    argv = [
        config["python_executable"],
        config["trainer"],
        "--config",
        config["promoted_config"],
        "--arm",
        arm,
        "--resume",
        "--device",
        "cuda",
    ]
    if smoke:
        argv.extend(
            [
                "--smoke-test",
                "--smoke-output-dir",
                str(summary_path.parent),
            ]
        )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["gpu"]["index"])
    result = command_runner(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise SupervisorError(
            f"{arm} {'smoke' if smoke else 'formal'} subprocess failed: "
            + result.stderr.strip()[-1000:]
        )
    record = _validate_arm_summary(config, arm=arm, smoke=smoke)
    record["argv"] = argv
    record["cuda_visible_devices"] = environment["CUDA_VISIBLE_DEVICES"]
    return record


def _record_wait(
    config: Mapping[str, Any],
    state: dict[str, Any],
    *,
    reason: str,
    snapshot: Mapping[str, Any],
) -> None:
    observations = state["wait_observations"]
    observations.append(
        {
            "observed_utc": utc_now(),
            "stage": state["current_stage"],
            "reason": reason,
            "snapshot": deepcopy(dict(snapshot)),
        }
    )
    if len(observations) > 512:
        del observations[:-512]
    _save_state(config, state)


def _fail(
    config: Mapping[str, Any],
    state: dict[str, Any],
    stage: str,
    error: BaseException,
) -> None:
    state["status"] = "failed"
    state["current_stage"] = stage
    state["failure"] = {
        "stage": stage,
        "failed_utc": utc_now(),
        "error_type": type(error).__name__,
        "message": str(error),
    }
    state["stages"][stage] = {
        "status": "failed",
        **state["failure"],
    }
    _save_state(config, state)


def _stage_success(
    config: Mapping[str, Any],
    state: dict[str, Any],
    stage: str,
    record: Mapping[str, Any],
) -> None:
    state["stages"][stage] = {
        "status": "succeeded",
        "completed_utc": utc_now(),
        "record": deepcopy(dict(record)),
    }
    index = STAGES.index(stage)
    state["current_stage"] = (
        STAGES[index + 1] if index + 1 < len(STAGES) else "complete"
    )
    _save_state(config, state)


def _revalidate_success(
    config: Mapping[str, Any],
    identity: Mapping[str, str],
    stage: str,
) -> dict[str, Any]:
    if stage == "prerequisites":
        selected = _selected_receipt_snapshot(config["selected_winner_receipt"])
        if not selected.get("ready"):
            raise SupervisorError("completed prerequisite receipt regressed")
        qwen_archive = _validate_archived_qwen_b_completion(
            config["selected_winner_receipt"]
        )
        video = _validate_video(config)
        return {
            "winner": selected,
            "qwen_b_archive": qwen_archive,
            "video": video,
        }
    if stage == "promotion":
        return _validate_promotion_outputs(config, identity)
    return _validate_arm_summary(
        config,
        arm=STAGE_ARM[stage],
        smoke=stage.endswith("_smoke"),
    )


def _write_final_receipt(
    config: Mapping[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RECEIPT_ARTIFACT_KIND,
        "status": "complete",
        "completed_utc": utc_now(),
        "config": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "state": config["state"],
        "state_sha256": sha256_file(config["state"]),
        "stage_order": list(STAGES),
        "stages": deepcopy(state["stages"]),
        "formal_arms_sequential": True,
        "arm_initialization": (
            "same_selected_winner_independent_no_cross_arm_warm_start"
        ),
        "gpu_policy": (
            "unknown_compute_record_only_never_kill_and_24GiB_free_"
            "required_for_two_consecutive_polls"
        ),
        "strict_hanyang_count": 344,
        "boundary_hanyang_admitted_count": 0,
        "kimodo_admitted_count": 0,
        "formal_release_eligible": False,
    }
    receipt["sha256"] = canonical_sha256(receipt)
    _atomic_json(receipt, config["receipt"])
    return receipt


def run_supervisor(
    config: Mapping[str, Any],
    *,
    approved_by: str,
    approved_utc: str,
    decision_notes: str,
    poll_seconds: float = 60.0,
    timeout_seconds: float | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if (
        not math.isfinite(poll_seconds)
        or poll_seconds <= 0
        or (
            timeout_seconds is not None
            and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0)
        )
    ):
        raise SupervisorError("poll/timeout must be positive")
    identity = _validate_identity(approved_by, approved_utc, decision_notes)
    state = _load_or_create_state(config, identity)
    started = time.monotonic()

    def wait(reason: str, snapshot: Mapping[str, Any]) -> None:
        _record_wait(config, state, reason=reason, snapshot=snapshot)
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= timeout_seconds
        ):
            raise SupervisorError(f"supervisor timed out while {reason}")
        sleeper(poll_seconds)

    for stage in STAGES:
        existing = state["stages"].get(stage)
        if isinstance(existing, Mapping) and existing.get("status") == "succeeded":
            try:
                existing["record"] = _revalidate_success(config, identity, stage)
                _save_state(config, state)
                continue
            except Exception as exc:
                _fail(config, state, stage, exc)
                raise
        state["current_stage"] = stage
        state["stages"][stage] = {
            "status": "running",
            "started_utc": utc_now(),
        }
        _save_state(config, state)
        try:
            if stage == "prerequisites":
                while True:
                    selected = _selected_receipt_snapshot(
                        config["selected_winner_receipt"]
                    )
                    qwen = _unit_snapshot(
                        config["qwen_b_training_service_unit"],
                        command_runner=command_runner,
                        allow_transient_reclaimed=True,
                    )
                    video_unit = _unit_snapshot(
                        config["video_waiter_service_unit"],
                        command_runner=command_runner,
                        allow_transient_reclaimed=True,
                    )
                    if qwen.get("failed") is True:
                        raise SupervisorError("Qwen B training unit failed")
                    if video_unit.get("failed") is True:
                        raise SupervisorError("GT/A/B video waiter unit failed")
                    if not selected.get("ready"):
                        wait("winner_selection_not_bindable_yet", selected)
                        continue
                    qwen_completion = _qwen_completion_snapshot(
                        qwen,
                        selected_receipt_path=config[
                            "selected_winner_receipt"
                        ],
                    )
                    if not qwen_completion["ready"]:
                        wait(
                            "qwen_b_training_unit_not_exited",
                            qwen_completion,
                        )
                        continue
                    video_completion = _video_completion_snapshot(
                        config, video_unit
                    )
                    if not video_completion["ready"]:
                        wait(
                            "video_waiter_unit_not_exited",
                            video_completion,
                        )
                        continue
                    record = {
                        "winner": selected,
                        "qwen_b_unit": qwen,
                        "video_waiter_unit": video_unit,
                        "qwen_b_completion": qwen_completion,
                        "video_completion": video_completion,
                        "video": video_completion["video"],
                    }
                    break
            elif stage == "promotion":
                record = _run_promotion(
                    config, identity, command_runner=command_runner
                )
            else:
                consecutive_memory_passes = 0
                gpu_observations: list[dict[str, Any]] = []
                while consecutive_memory_passes < 2:
                    gpu = _gpu_snapshot(
                        config, command_runner=command_runner
                    )
                    gpu_observations.append(gpu)
                    if gpu["ready"]:
                        consecutive_memory_passes += 1
                    else:
                        consecutive_memory_passes = 0
                    if consecutive_memory_passes < 2:
                        wait(
                            (
                                "gpu_free_memory_stability_confirmation"
                                if gpu["ready"]
                                else "gpu_free_memory_below_24GiB"
                            ),
                            {
                                **gpu,
                                "consecutive_memory_passes": (
                                    consecutive_memory_passes
                                ),
                                "required_consecutive_memory_passes": 2,
                            },
                        )
                record = _run_arm(
                    config,
                    arm=STAGE_ARM[stage],
                    smoke=stage.endswith("_smoke"),
                    command_runner=command_runner,
                )
                record["prelaunch_gpu_gate"] = {
                    "passed": True,
                    "required_consecutive_memory_passes": 2,
                    "observations": gpu_observations,
                    "unknown_compute_policy": "record_only_never_kill",
                }
            _stage_success(config, state, stage, record)
        except Exception as exc:
            _fail(config, state, stage, exc)
            if isinstance(exc, SupervisorError):
                raise
            raise SupervisorError(f"{stage} failed") from exc
    state["status"] = "complete"
    state["current_stage"] = "complete"
    _save_state(config, state)
    return _write_final_receipt(config, state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs/hanyang_beat2_emotion_preserving_v81_night_supervisor.json"
        ),
    )
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-utc", required=True)
    parser.add_argument("--decision-notes", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = read_config(args.config)
    lock_path = Path(config["state"] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("ERROR: another night supervisor holds the lock", file=sys.stderr)
            return 3
        try:
            receipt = run_supervisor(
                config,
                approved_by=args.approved_by,
                approved_utc=args.approved_utc,
                decision_notes=args.decision_notes,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
