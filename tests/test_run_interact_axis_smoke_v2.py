import hashlib
import json

from tools.gmr_v2.run_interact_axis_smoke_v2 import (
    output_paths,
    validate_reuse_source_state,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reused_retarget_state_binds_quality_and_safe_csv_hashes(tmp_path):
    output_root = tmp_path / "processed"
    review_root = tmp_path / "review"
    output_root.mkdir()
    task_id = "interact_task_001"
    paths = output_paths(task_id, output_root, review_root)
    paths["quality"].write_text("{}\n", encoding="utf-8")
    paths["safe_csv"].write_text("joint\n0\n", encoding="utf-8")
    receipt_hash = "a" * 64
    state = {
        "artifact_kind": "interact_native_bvh_axis_smoke_v2_run_state",
        "status": "complete_pending_blind_review",
        "failure_count": 0,
        "input_receipt_sha256": receipt_hash,
        "code_sha256": {"retarget": "b" * 64},
        "results": {
            task_id: {
                "status": "rendered_pending_blind_review",
                "artifacts": {
                    "quality": {
                        "path": str(paths["quality"]),
                        "sha256": _sha(paths["quality"]),
                    },
                    "safe_csv": {
                        "path": str(paths["safe_csv"]),
                        "sha256": _sha(paths["safe_csv"]),
                    },
                },
            }
        },
    }
    state_path = tmp_path / "source_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    binding = validate_reuse_source_state(
        state_path,
        receipt_hash=receipt_hash,
        rows=[{"episode_task_id": task_id}],
        output_root=output_root,
    )
    assert binding["physical_retarget_code_sha256"] == "b" * 64
    assert binding["all_quality_and_safe_csv_hashes_verified"] is True
