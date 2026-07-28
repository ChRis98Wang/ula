"""Closed data-source registry and lineage guards for motion training.

The registry separates *being known* from *being admitted to a training role*.
In particular, the Hanyang/Duksung emotional-body dataset is registered for an
isolated emotion critic/calibration lane, but it is not an admissible generator
foundation source.  Kimodo is permanently denied at every provenance layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import re
from typing import Any


DATA_SOURCE_REGISTRY_CONTRACT_TYPE = "ula_motion_data_source_registry_v1"
DATA_SOURCE_REGISTRY_CONTRACT_VERSION = 1
DATA_SOURCE_REGISTRY_HASH_FIELD = "data_source_registry_sha256"

BEAT2_FORMAL_SOURCE_ID = "beat2_official_semantic_event_training_pool_v7"
BEAT2_FORMAL_V8_EXPANDED_SOURCE_ID = (
    "beat2_official_semantic_event_training_pool_v8_expanded"
)
BEAT2_INTERACTION_SOURCE_ID = "beat2_interaction_18d_v1"
BEAT2_EXPRESSION_TURN_SOURCE_ID = "beat2_expression_turn_v8"
HANYANG_EMOTIONAL_BODY_SOURCE_ID = (
    "hanyang_duksung_emotional_body_motion_v1"
)
ULA0513_USER_OWNED_SOURCE_ID = "ula0513_user_owned"

GENERATOR_FOUNDATION_ROLE = "generator_foundation"
SEMANTIC_GENERATOR_ROLE = "semantic_generator"
EXPRESSION_GENERATOR_ROLE = "expression_generator"
EMOTION_CRITIC_ROLE = "emotion_critic"
EMOTION_CALIBRATION_ROLE = "emotion_calibration"
EMOTION_EVALUATION_ROLE = "emotion_evaluation"

KIMODO_PERMANENT_DENY_POLICY = (
    "kimodo_permanent_hard_deny_raw_cache_normalizer_split_checkpoint_v1"
)

_GENERATOR_ROLES = frozenset(
    {
        GENERATOR_FOUNDATION_ROLE,
        SEMANTIC_GENERATOR_ROLE,
        EXPRESSION_GENERATOR_ROLE,
    }
)
_HANYANG_ROLES = frozenset(
    {
        EMOTION_CRITIC_ROLE,
        EMOTION_CALIBRATION_ROLE,
        EMOTION_EVALUATION_ROLE,
    }
)

# This is deliberately an exact-ID registry.  Dataset-family prefixes are not
# accepted as registration, because an arbitrary string such as
# ``beat2_something`` is not provenance.
DATA_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    BEAT2_FORMAL_SOURCE_ID: {
        "dataset_family": "BEAT2",
        "isolation_domain": "beat2_generator",
        "allowed_roles": sorted(_GENERATOR_ROLES | _HANYANG_ROLES),
        "robot_retarget_required": True,
        "license_scope": "non-commercial_research_only",
    },
    BEAT2_FORMAL_V8_EXPANDED_SOURCE_ID: {
        "dataset_family": "BEAT2",
        "isolation_domain": "beat2_generator",
        "allowed_roles": sorted(_GENERATOR_ROLES | _HANYANG_ROLES),
        "robot_retarget_required": True,
        "license_scope": "non-commercial_research_only",
    },
    BEAT2_INTERACTION_SOURCE_ID: {
        "dataset_family": "BEAT2",
        "isolation_domain": "beat2_generator",
        "allowed_roles": sorted(_GENERATOR_ROLES | _HANYANG_ROLES),
        "robot_retarget_required": True,
        "license_scope": "non-commercial_research_only",
    },
    BEAT2_EXPRESSION_TURN_SOURCE_ID: {
        "dataset_family": "BEAT2",
        "isolation_domain": "beat2_expression_generator",
        "allowed_roles": sorted(
            {SEMANTIC_GENERATOR_ROLE, EXPRESSION_GENERATOR_ROLE}
            | _HANYANG_ROLES
        ),
        "robot_retarget_required": True,
        "license_scope": "non-commercial_research_only",
    },
    ULA0513_USER_OWNED_SOURCE_ID: {
        "dataset_family": "ULA_USER_OWNED",
        "isolation_domain": "user_owned_generator",
        "allowed_roles": sorted(
            {SEMANTIC_GENERATOR_ROLE, EXPRESSION_GENERATOR_ROLE}
            | _HANYANG_ROLES
        ),
        "robot_retarget_required": True,
        "license_scope": "explicit_user_owned_authorization",
    },
    HANYANG_EMOTIONAL_BODY_SOURCE_ID: {
        "dataset_family": "HANYANG_DUKSUNG_EMOTIONAL_BODY_MOTION",
        "isolation_domain": "external_emotion_critic_only",
        "allowed_roles": sorted(_HANYANG_ROLES),
        "robot_retarget_required": True,
        "license_scope": "cc-by-4.0",
        "generator_foundation_allowed": False,
        "label_quality_policy": (
            "observer_agreement_filtered_critic_or_calibration_only"
        ),
    },
}

_FORBIDDEN_SOURCE_TOKENS = ("kimodo",)
_SOURCE_BEARING_EXACT_KEYS = frozenset(
    {
        "cache",
        "cache_path",
        "checkpoint",
        "checkpoint_path",
        "clip_id",
        "dataset",
        "dataset_family",
        "dataset_id",
        "dataset_name",
        "dataset_source",
        "generator_checkpoint",
        "manifest",
        "normalizer",
        "normalizer_path",
        "origin",
        "origin_dataset",
        "path",
        "raw_path",
        "raw_root",
        "source",
        "source_clip_id",
        "source_dataset",
        "source_group_key",
        "source_inventory",
        "source_manifest",
        "source_name",
        "source_path",
        "split",
        "split_path",
        "trajectory_path",
        "uri",
        "url",
    }
)
_SOURCE_BEARING_SUFFIXES = (
    "_cache",
    "_checkpoint",
    "_dataset",
    "_dir",
    "_lock",
    "_manifest",
    "_path",
    "_relpath",
    "_root",
    "_source",
    "_uri",
    "_url",
)
_POLICY_OR_SCHEMA_KEY_PARTS = (
    "policy",
    "condition",
    "emotion",
    "behavior",
    "label",
    "layout",
    "supervision",
    "ontology",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _contract_sha256(value: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: item for key, item in value.items() if key != "sha256"}
    )


def _normalized_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _forbidden_token(value: object) -> str | None:
    normalized = _normalized_token(value)
    return next(
        (token for token in _FORBIDDEN_SOURCE_TOKENS if token in normalized),
        None,
    )


def assert_no_forbidden_source_reference(
    value: object,
    *,
    context: str,
) -> None:
    """Reject Kimodo when ``value`` is explicitly source-bearing."""

    token = _forbidden_token(value)
    if token is not None:
        raise ValueError(
            f"{context} references permanently forbidden dataset {token!r}"
        )


def _is_source_bearing_key(key: object) -> bool:
    name = str(key).strip().casefold()
    if _is_policy_or_schema_key(name):
        return False
    return name in _SOURCE_BEARING_EXACT_KEYS or name.endswith(
        _SOURCE_BEARING_SUFFIXES
    )


def _is_policy_or_schema_key(key: object) -> bool:
    name = str(key).strip().casefold()
    return any(part in name for part in _POLICY_OR_SCHEMA_KEY_PARTS)


def assert_no_forbidden_data_lineage(
    payload: object,
    *,
    context: str,
    _source_bearing: bool = False,
) -> None:
    """Walk provenance payloads and reject forbidden source/path identifiers.

    Policy and condition-schema strings are intentionally not treated as data
    lineage.  This lets a checkpoint state the no-Kimodo policy while still
    rejecting a path, dataset ID, clip ID, cache, split, normalizer, or
    checkpoint that actually points at Kimodo.
    """

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_context = f"{context}.{key}"
            child_source_bearing = _is_source_bearing_key(key) or (
                _source_bearing and not _is_policy_or_schema_key(key)
            )
            assert_no_forbidden_data_lineage(
                value,
                context=child_context,
                _source_bearing=child_source_bearing,
            )
        return
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        for index, value in enumerate(payload):
            assert_no_forbidden_data_lineage(
                value,
                context=f"{context}[{index}]",
                _source_bearing=_source_bearing,
            )
        return
    if _source_bearing and isinstance(payload, (str, bytes)):
        assert_no_forbidden_source_reference(payload, context=context)


def registered_source(
    dataset_source: object,
    *,
    role: str,
) -> dict[str, Any]:
    """Return an immutable registration view or fail closed."""

    source_id = str(dataset_source or "").strip().casefold()
    if not source_id:
        raise ValueError("dataset_source must be explicit")
    assert_no_forbidden_source_reference(source_id, context="dataset_source")
    registration = DATA_SOURCE_REGISTRY.get(source_id)
    if registration is None:
        raise ValueError(
            f"unregistered dataset_source {source_id!r}; expected one of "
            f"{sorted(DATA_SOURCE_REGISTRY)}"
        )
    if role not in registration["allowed_roles"]:
        raise ValueError(
            f"dataset_source {source_id!r} is not admitted for role {role!r}; "
            f"allowed roles are {registration['allowed_roles']}"
        )
    return {"dataset_source": source_id, **deepcopy(registration)}


def build_data_source_registry_contract(
    dataset_sources: Sequence[object],
    *,
    role: str,
) -> dict[str, Any]:
    """Build the hash-bound registry snapshot carried by derived artifacts."""

    source_ids = sorted(
        {
            str(value or "").strip().casefold()
            for value in dataset_sources
        }
    )
    if not source_ids or "" in source_ids:
        raise ValueError("at least one explicit dataset_source is required")
    registrations = [
        registered_source(source_id, role=role) for source_id in source_ids
    ]
    payload: dict[str, Any] = {
        "contract_type": DATA_SOURCE_REGISTRY_CONTRACT_TYPE,
        "contract_version": DATA_SOURCE_REGISTRY_CONTRACT_VERSION,
        "use_role": role,
        "deny_policy": KIMODO_PERMANENT_DENY_POLICY,
        "forbidden_dataset_ids": list(_FORBIDDEN_SOURCE_TOKENS),
        "sources": registrations,
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def validate_data_source_registry_contract(
    contract: object,
    *,
    expected_role: str | None = None,
    expected_dataset_sources: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Validate a registry snapshot against the closed in-code registry."""

    if not isinstance(contract, Mapping):
        raise ValueError("data-source registry contract is missing")
    if (
        contract.get("contract_type") != DATA_SOURCE_REGISTRY_CONTRACT_TYPE
        or contract.get("contract_version")
        != DATA_SOURCE_REGISTRY_CONTRACT_VERSION
        or contract.get("deny_policy") != KIMODO_PERMANENT_DENY_POLICY
        or contract.get("forbidden_dataset_ids")
        != list(_FORBIDDEN_SOURCE_TOKENS)
        or contract.get("sha256") != _contract_sha256(contract)
    ):
        raise ValueError("data-source registry contract is invalid or changed")
    role = str(contract.get("use_role") or "")
    if expected_role is not None and role != expected_role:
        raise ValueError(
            "data-source registry role changed: "
            f"expected {expected_role!r}, got {role!r}"
        )
    sources = contract.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data-source registry contract has no sources")
    expected = [
        registered_source(record.get("dataset_source"), role=role)
        if isinstance(record, Mapping)
        else None
        for record in sources
    ]
    if sources != expected:
        raise ValueError(
            "data-source registry snapshot differs from the closed registry"
        )
    source_ids = [record["dataset_source"] for record in expected]
    if source_ids != sorted(set(source_ids)):
        raise ValueError("data-source registry sources must be unique and sorted")
    if expected_dataset_sources is not None and source_ids != sorted(
        {str(value or "").strip().casefold() for value in expected_dataset_sources}
    ):
        raise ValueError("artifact dataset sources differ from its registry snapshot")
    assert_no_forbidden_data_lineage(contract, context="data_source_registry")
    return dict(contract)


def bind_contract_to_data_sources(
    contract: Mapping[str, Any],
    registry_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash a JSON contract after binding it to the source registry."""

    validated = validate_data_source_registry_contract(registry_contract)
    payload = {
        key: value
        for key, value in contract.items()
        if key not in {"sha256", DATA_SOURCE_REGISTRY_HASH_FIELD}
    }
    payload[DATA_SOURCE_REGISTRY_HASH_FIELD] = validated["sha256"]
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def validate_contract_source_binding(
    contract: object,
    registry_contract: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Require a split/normalizer/cache contract to bind the same registry."""

    if not isinstance(contract, Mapping):
        raise ValueError(f"{context} contract is missing")
    registry = validate_data_source_registry_contract(registry_contract)
    if contract.get(DATA_SOURCE_REGISTRY_HASH_FIELD) != registry["sha256"]:
        raise ValueError(f"{context} is not bound to the data-source registry")
    assert_no_forbidden_data_lineage(contract, context=context)


__all__ = [
    "BEAT2_FORMAL_SOURCE_ID",
    "BEAT2_EXPRESSION_TURN_SOURCE_ID",
    "BEAT2_INTERACTION_SOURCE_ID",
    "DATA_SOURCE_REGISTRY",
    "DATA_SOURCE_REGISTRY_CONTRACT_TYPE",
    "DATA_SOURCE_REGISTRY_HASH_FIELD",
    "EMOTION_CALIBRATION_ROLE",
    "EMOTION_CRITIC_ROLE",
    "EMOTION_EVALUATION_ROLE",
    "EXPRESSION_GENERATOR_ROLE",
    "GENERATOR_FOUNDATION_ROLE",
    "HANYANG_EMOTIONAL_BODY_SOURCE_ID",
    "KIMODO_PERMANENT_DENY_POLICY",
    "SEMANTIC_GENERATOR_ROLE",
    "ULA0513_USER_OWNED_SOURCE_ID",
    "assert_no_forbidden_data_lineage",
    "assert_no_forbidden_source_reference",
    "bind_contract_to_data_sources",
    "build_data_source_registry_contract",
    "registered_source",
    "validate_contract_source_binding",
    "validate_data_source_registry_contract",
]
