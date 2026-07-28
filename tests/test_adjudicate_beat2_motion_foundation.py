import copy

import pytest

from tools import adjudicate_beat2_motion_foundation as foundation
from tools.gmr_v2 import batch_retarget_beat2_v2 as ordinary


def _source_record():
    gates = {name: True for name in foundation.REQUIRED_GATES}
    return {
        "clip_id": "foundation_chunk",
        "task_id": "foundation_chunk",
        "dataset": "BEAT2",
        "dataset_subset": "beat_english_v2.0.0",
        "annotation_kind": foundation.FOUNDATION_ANNOTATION_KIND,
        "speaker_key": "1_wayne",
        "fixed_split_assignment": "train",
        "source_group_key": "BEAT2/beat_english_v2.0.0/1_wayne_0_1_1",
        "semantic_label_status": "absent_motion_foundation",
        "semantic_supervision_masks": dict(foundation.SEMANTIC_MASKS),
        "behavior_supervision_mask": False,
        "emotion_supervision_mask": False,
        "affect_observable_supervision_mask": False,
        "official_category_conditioning_enabled": False,
        "official_emotion_conditioning_enabled": False,
        "training_segment": {
            "representation": foundation.FOUNDATION_REPRESENTATION,
            "fixed_window_sec": None,
            "frame_count": 300,
            "overlap_frames": 0,
            "boundary_source": "source_container_frame_bounds",
        },
        "retarget_segment": {
            "representation": (
                foundation.MOTION_FOUNDATION_RETARGET_SEGMENT_REPRESENTATION
            ),
            "cropped": False,
            "output_frame_count": 310,
        },
        "frames": 310,
        "fps": 30.0,
        "quality_gate": gates,
        "quality_json": "/BEAT2/quality.json",
        "quality_json_sha256": "a" * 64,
        "safe_csv": "/BEAT2/safe.csv",
        "safe_csv_sha256": "b" * 64,
        "lineage_hashes": {},
        "accepted_for_training": False,
        "audio_relpath": None,
        "source_speech_context": "",
        "semantic_admission": "not_applicable_unlabeled_motion_foundation",
    }


def test_adjudicated_record_strips_conditioning_and_stays_unadmitted():
    source = _source_record()

    result = foundation._adjudicated_record(source, smoke_only=False)

    assert result["accepted_for_training"] is False
    assert result["adjudication"]["training_admitted"] is False
    assert result["motion_only_admission"]["text_conditioning_enabled"] is False
    assert result["motion_18d"]["frames"] == 310
    assert not foundation.STRIP_FROM_ADJUDICATED.intersection(result)


def test_run_contract_requires_the_original_qc_policy_and_reviewed_parameters():
    contract = {
        "quality_policy": dict(ordinary.QUALITY_POLICY),
        "output_contract": ordinary.ULA_V2_18D_CONTRACT,
        "axis_policy": ordinary.BEAT2_AXIS_POLICY,
        "retarget_parameters": {
            **ordinary.RETARGET_PARAMETERS,
            "neutral_limit_margin_rad": 1e-6,
        },
    }
    digest = ordinary.json_sha256(contract)
    saved = {"run_contract": contract, "run_contract_sha256": digest}
    status = {
        "run_contract": contract,
        "run_contract_sha256": digest,
        "inventory_sha256": "c" * 64,
    }

    returned, returned_hash = foundation._validate_run_contract(
        status=status,
        saved=saved,
        inventory_sha256="c" * 64,
    )

    assert returned == contract
    assert returned_hash == digest

    changed = copy.deepcopy(contract)
    changed["quality_policy"]["limb_target_error_p95_max_m"] = 0.05
    changed_digest = ordinary.json_sha256(changed)
    with pytest.raises(ValueError, match="thresholds changed"):
        foundation._validate_run_contract(
            status={
                **status,
                "run_contract": changed,
                "run_contract_sha256": changed_digest,
            },
            saved={
                "run_contract": changed,
                "run_contract_sha256": changed_digest,
            },
            inventory_sha256="c" * 64,
        )
