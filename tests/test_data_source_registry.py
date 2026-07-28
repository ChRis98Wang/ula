import json

import pytest

from tools.init_ula_v2_18d_random import load_formal_sources
from upper_body_skeleton.data_source_registry import (
    BEAT2_FORMAL_SOURCE_ID,
    DATA_SOURCE_REGISTRY_HASH_FIELD,
    EMOTION_CRITIC_ROLE,
    GENERATOR_FOUNDATION_ROLE,
    HANYANG_EMOTIONAL_BODY_SOURCE_ID,
    assert_no_forbidden_data_lineage,
    bind_contract_to_data_sources,
    build_data_source_registry_contract,
    registered_source,
    validate_contract_source_binding,
    validate_data_source_registry_contract,
)
from upper_body_skeleton.ula_v2_18d_head import MOTION_ONLY_EPISODE_CONTRACT


def test_registry_is_exact_and_hanyang_is_critic_only():
    beat2 = registered_source(
        BEAT2_FORMAL_SOURCE_ID, role=GENERATOR_FOUNDATION_ROLE
    )
    assert beat2["dataset_family"] == "BEAT2"

    hanyang = registered_source(
        HANYANG_EMOTIONAL_BODY_SOURCE_ID, role=EMOTION_CRITIC_ROLE
    )
    assert hanyang["isolation_domain"] == "external_emotion_critic_only"
    assert hanyang["generator_foundation_allowed"] is False

    with pytest.raises(ValueError, match="not admitted"):
        registered_source(
            HANYANG_EMOTIONAL_BODY_SOURCE_ID,
            role=GENERATOR_FOUNDATION_ROLE,
        )
    with pytest.raises(ValueError, match="unregistered"):
        registered_source(
            "beat2_arbitrary_string",
            role=GENERATOR_FOUNDATION_ROLE,
        )
    with pytest.raises(ValueError, match="permanently forbidden"):
        registered_source("Ki-mo-do", role=EMOTION_CRITIC_ROLE)


@pytest.mark.parametrize(
    ("layer", "payload"),
    [
        ("raw", {"source": "/datasets/Ki-mo-do/raw/clip.npz"}),
        ("nested_raw", {"source": {"name": "KIMODO"}}),
        ("cache", {"cache_path": "/runs/kimodo_cache/conditions.npz"}),
        ("normalizer", {"normalizer_path": "/stats/KIMODO/style.json"}),
        ("split", {"dataset_source": "kimodo_train_split"}),
        ("checkpoint", {"checkpoint_path": "/models/kimodo/base.pt"}),
        ("provenance_lock", {"source_provenance_lock": "/locks/kimodo.json"}),
    ],
)
def test_kimodo_is_hard_denied_at_every_lineage_layer(layer, payload):
    with pytest.raises(ValueError, match="permanently forbidden"):
        assert_no_forbidden_data_lineage(payload, context=layer)


def test_registry_hash_binds_split_and_normalizer_contracts():
    registry = build_data_source_registry_contract(
        [BEAT2_FORMAL_SOURCE_ID],
        role=GENERATOR_FOUNDATION_ROLE,
    )
    validate_data_source_registry_contract(
        registry,
        expected_role=GENERATOR_FOUNDATION_ROLE,
        expected_dataset_sources=[BEAT2_FORMAL_SOURCE_ID],
    )

    for name in ("split", "action_normalizer", "style_normalizer"):
        bound = bind_contract_to_data_sources(
            {"contract_type": name, "contract_version": 1},
            registry,
        )
        assert bound[DATA_SOURCE_REGISTRY_HASH_FIELD] == registry["sha256"]
        validate_contract_source_binding(bound, registry, context=name)

        tampered = dict(bound)
        tampered[DATA_SOURCE_REGISTRY_HASH_FIELD] = "0" * 64
        with pytest.raises(ValueError, match="not bound"):
            validate_contract_source_binding(
                tampered, registry, context=name
            )


def test_manifest_cannot_hide_kimodo_under_registered_beat2_source(tmp_path):
    manifest = tmp_path / "train_ready.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "masked-origin",
                "source": "/private/KIMODO/raw/masked-origin.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "formal_episode_contract": MOTION_ONLY_EPISODE_CONTRACT,
        "motion_sources": [
            {
                "dataset_source": BEAT2_FORMAL_SOURCE_ID,
                "manifest": str(manifest),
                "speaker_namespace": "beat2",
                "source_group_namespace": "beat2",
                "use_manifest_fixed_split": True,
            }
        ],
    }

    with pytest.raises(ValueError, match="permanently forbidden"):
        load_formal_sources(config)
