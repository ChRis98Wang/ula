import copy

from tools.human_motion_review.adjudicate_training_dataset import REQUIRED_REVIEW_GATES
from tools.human_motion_review.validate_review_bundle import (
    REQUIRED_GATES,
    review_gate_template,
    validate_bundle,
)


def _fixture():
    evidence = {
        key: {"path": f"/{key}", "sha256": "0" * 64}
        for key in ("preview", "raw_label", "quality_report")
    }
    sample = {
        "sample_id": "sample_1",
        "status": "agent_reviewed",
        "scores": {"recognizability": 0.9, "text_consistency": 0.8, "observable_dof_coverage": 1.0},
        "gates": {
            gate: gate != "affect_observable_in_18d"
            for gate in review_gate_template()
        },
        "split": {
            "subject_key": None,
            "subject_policy": "train_only_unknown",
            "action_key": "wave",
            "source_group_key": "source/1",
            "assignment": "train",
            "eval_eligible": False,
        },
        "training_acceptance": True,
        "conflict_ids": [],
        "evidence": evidence,
    }
    review = {
        "schema_version": 1,
        "review_id": "review_1",
        "reviewer": {"kind": "agent", "independent_of_annotation_logic": True},
        "samples": [sample],
        "summary": {"status_counts": {"agent_reviewed": 1}, "training_accepted": 1},
    }
    conflicts = {"schema_version": 1, "review_id": "review_1", "items": []}
    return review, conflicts


def test_valid_train_only_unknown_subject_bundle():
    review, conflicts = _fixture()
    assert validate_bundle(review, conflicts) == []


def test_validator_and_adjudicator_share_the_exact_18d_gate_contract():
    assert REQUIRED_GATES == REQUIRED_REVIEW_GATES
    assert set(review_gate_template()) == REQUIRED_GATES
    assert review_gate_template()["observable_in_18d"] is False
    assert "observable_in_15d" not in review_gate_template()


def test_legacy_15d_bundle_fails_closed_without_silent_migration():
    review, conflicts = _fixture()
    gates = review["samples"][0]["gates"]
    gates["observable_in_15d"] = gates.pop("observable_in_18d")

    errors = validate_bundle(review, conflicts)

    assert any(
        "legacy observable_in_15d cannot validate an 18D review" in error
        for error in errors
    )
    assert "observable_in_18d" not in gates


def test_status_is_closed_enum():
    review, conflicts = _fixture()
    review["samples"][0]["status"] = "approved"
    assert any("invalid status" in error for error in validate_bundle(review, conflicts))


def test_acceptance_requires_every_motion_admission_gate():
    review, conflicts = _fixture()
    review["samples"][0]["gates"]["text_consistent"] = False
    assert any(
        "pass every motion-admission gate" in error
        for error in validate_bundle(review, conflicts)
    )


def test_affect_gate_does_not_block_motion_acceptance_when_false():
    review, conflicts = _fixture()
    assert review["samples"][0]["gates"]["affect_observable_in_18d"] is False
    assert validate_bundle(review, conflicts) == []


def test_affect_gate_requires_bound_blind_review_evidence_when_true():
    review, conflicts = _fixture()
    review["samples"][0]["gates"]["affect_observable_in_18d"] = True
    errors = validate_bundle(review, conflicts)
    assert any("requires blind review evidence" in error for error in errors)


def test_unknown_subject_cannot_enter_evaluation():
    review, conflicts = _fixture()
    review["samples"][0]["training_acceptance"] = False
    review["summary"]["training_accepted"] = 0
    review["samples"][0]["split"]["assignment"] = "test"
    assert any("unknown subject cannot enter" in error for error in validate_bundle(review, conflicts))


def test_source_group_cannot_cross_splits():
    review, conflicts = _fixture()
    second = copy.deepcopy(review["samples"][0])
    second["sample_id"] = "sample_2"
    second["training_acceptance"] = False
    second["split"]["subject_key"] = "subject_2"
    second["split"]["assignment"] = "val"
    second["split"]["eval_eligible"] = True
    review["samples"].append(second)
    review["summary"]["status_counts"]["agent_reviewed"] = 2
    assert any("source group" in error and "crosses splits" in error for error in validate_bundle(review, conflicts))


def test_ood_action_must_be_disjoint_from_train_actions():
    review, conflicts = _fixture()
    second = copy.deepcopy(review["samples"][0])
    second["sample_id"] = "sample_2"
    second["training_acceptance"] = False
    second["split"].update(
        {
            "subject_key": "subject_2",
            "source_group_key": "source/2",
            "assignment": "ood_action_test",
            "eval_eligible": True,
        }
    )
    review["samples"].append(second)
    review["summary"]["status_counts"]["agent_reviewed"] = 2
    assert any("OOD action" in error for error in validate_bundle(review, conflicts))


def test_conflict_must_reference_its_sample():
    review, conflicts = _fixture()
    review["samples"][0]["conflict_ids"] = ["conflict_1"]
    conflicts["items"] = [
        {
            "conflict_id": "conflict_1",
            "scope": "sample",
            "sample_id": "other",
            "status": "needs_human",
            "severity": "blocking",
        }
    ]
    errors = validate_bundle(copy.deepcopy(review), conflicts)
    assert any("unknown sample_id" in error or "another sample" in error for error in errors)
