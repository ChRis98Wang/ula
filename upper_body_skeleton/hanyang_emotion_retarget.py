"""Contracts and geometry for the Hanyang emotional-body-motion dataset.

The official source contains global XYZ positions only.  It therefore supports
position-constrained body IK, but it does not observe wrist twist/pitch or head
yaw.  Those five ULA dimensions are always exported with an explicit zero loss
mask; callers must never silently promote them to motion ground truth.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from upper_body_skeleton.data_source_registry import (
    HANYANG_EMOTIONAL_BODY_SOURCE_ID,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D


DATASET = "Hanyang/Duksung Emotional Body Motion"
DATASET_ID = HANYANG_EMOTIONAL_BODY_SOURCE_ID
DATASET_REVISION = "zenodo_record_10052504_v1"
DATASET_LICENSE = "CC-BY-4.0"
ZENODO_RECORD_URL = "https://zenodo.org/records/10052504"
SOURCE_ARCHIVE_NAME = "Emotional Body Motion Data.zip"
SOURCE_ARCHIVE_BYTES = 304_786_747
SOURCE_ARCHIVE_MD5 = "03965a5bb33369c5ba22338fd4ca783a"
HUMAN_EVALUATION_NAME = "Human Evaluation result.xlsx"
HUMAN_EVALUATION_BYTES = 743_483
HUMAN_EVALUATION_MD5 = "895e5523036700ae04e1c76dae1a1c9b"
SOURCE_FPS = 30.0
SOURCE_FRAMES = 150
SOURCE_CLIP_COUNT = 4_060
PARTICIPANT_COUNT = 29
RATER_COUNT = 13
MIN_VALID_RATERS = 1
HUMAN_RELIABILITY_MIN_RATERS = 10
HUMAN_CONFIDENCE_THRESHOLD = 0.70
SOURCE_BONE_CV_LIMIT = 0.05
EVALUATION_PROTOCOL_ANOMALY_STEMS = frozenset(
    {
        "15_2_3_1",
        "20_2_2_2",
    }
)
EXACT_DUPLICATE_STEM_GROUPS = (
    ("15_2_3_1", "15_2_4_1"),
    ("15_3_3_2", "15_3_4_2"),
    ("29_2_2_7", "29_2_3_7"),
    ("9_1_4_1", "9_3_4_1"),
)
# 15_2_3_1 is already excluded for a human-evaluation protocol anomaly, so
# its otherwise-identical peer 15_2_4_1 remains the usable representative.
DUPLICATE_EXCLUDED_STEMS = frozenset(
    {
        "15_3_4_2",
        "29_2_3_7",
        "9_3_4_1",
    }
)

EMOTION_BY_ID = {
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "angry",
    5: "disgust",
    6: "fear",
    7: "neutral",
}
EMOTION_ID_BY_NAME = {value: key for key, value in EMOTION_BY_ID.items()}

SOURCE_JOINTS = (
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
)
UPPER_BODY_SOURCE_JOINTS = SOURCE_JOINTS[:13]
SOURCE_HEADER = (
    "Frame",
    *(
        f"{joint}.{axis}"
        for joint in SOURCE_JOINTS
        for axis in ("x", "y", "z")
    ),
)

# Source global axes are x=anatomical right, y=up, z=front.  ULA/MuJoCo uses
# x=front, y=right, z=up.  This is a proper rotation, not a reflection.
SOURCE_TO_ROBOT_BASIS = np.asarray(
    (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)
AXIS_POLICY = "hanyang_global_x_right_y_up_z_front_to_ula_x_front_y_right_z_up_v1"

UNOBSERVED_18D_JOINTS = frozenset(
    {
        "joint_lWristRoll",
        "joint_lWristPitch",
        "joint_rWristRoll",
        "joint_rWristPitch",
        "head_yaw_joint",
    }
)
ACTION_DIM_MASK_18D = tuple(
    joint not in UNOBSERVED_18D_JOINTS for joint in JOINT_ORDER_18D
)
HEAD_OBSERVATION_POLICY = (
    "neck_to_head_direction_observes_roll_pitch_minimum_rotation_yaw_masked"
)
WRIST_OBSERVATION_POLICY = "hand_point_has_no_orientation_wrist_dofs_masked"

DEFAULT_PARTICIPANT_SPLIT = {
    "train": tuple(range(1, 22)),
    "validation": tuple(range(22, 26)),
    "test": tuple(range(26, 30)),
}
SPLITS = tuple(DEFAULT_PARTICIPANT_SPLIT)
FORBIDDEN_DATASET_MARKERS = ("kimodo",)
CLIP_NAME = re.compile(
    r"^(?P<participant>[1-9][0-9]*)_"
    r"(?P<block>[1-9][0-9]*)_"
    r"(?P<trial>[1-9][0-9]*)_"
    r"(?P<emotion>[1-7])(?:\.csv)?$",
    re.IGNORECASE,
)

_XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_hash(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_dataset_marker(value: object, *, context: str = "$") -> None:
    """Reject Kimodo at every Hanyang configuration/provenance ingress."""
    if isinstance(value, str):
        folded = value.casefold()
        if any(marker in folded for marker in FORBIDDEN_DATASET_MARKERS):
            raise ValueError(f"{context} contains a forbidden dataset marker")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            reject_forbidden_dataset_marker(str(key), context=f"{context}.<key>")
            reject_forbidden_dataset_marker(child, context=f"{context}.{key}")
        return
    if isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            reject_forbidden_dataset_marker(child, context=f"{context}[{index}]")


def fixed_split_for_participant(participant_id: int) -> str:
    participant_id = int(participant_id)
    for split, participants in DEFAULT_PARTICIPANT_SPLIT.items():
        if participant_id in participants:
            return split
    raise ValueError(f"participant outside official Hanyang range: {participant_id}")


def parse_clip_name(value: str | Path) -> dict[str, Any]:
    name = Path(value).name
    match = CLIP_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid Hanyang clip filename: {name!r}")
    participant = int(match.group("participant"))
    block = int(match.group("block"))
    trial = int(match.group("trial"))
    emotion_index = int(match.group("emotion"))
    if not 1 <= participant <= PARTICIPANT_COUNT:
        raise ValueError(f"Hanyang participant out of range: {participant}")
    if not 1 <= block <= 4:
        raise ValueError(f"Hanyang block out of range: {block}")
    if not 1 <= trial <= 5:
        raise ValueError(f"Hanyang trial out of range: {trial}")
    if not 1 <= emotion_index <= len(EMOTION_BY_ID):
        raise ValueError(f"Hanyang emotion out of range: {emotion_index}")
    stem = f"{participant}_{block}_{trial}_{emotion_index}"
    return {
        "clip_id": f"hanyang:{stem}",
        "source_stem": stem,
        "participant_id": participant,
        "block_id": block,
        "trial_id": trial,
        "emotion_index": emotion_index,
        "emotion_id": EMOTION_BY_ID[emotion_index],
        "fixed_split_assignment": fixed_split_for_participant(participant),
        "speaker_key": f"hanyang:participant:{participant:02d}",
        "source_group_key": f"hanyang:participant:{participant:02d}",
    }


def validate_official_file(
    path: str | Path, *, expected_bytes: int, expected_md5: str
) -> dict[str, Any]:
    path = Path(path).resolve()
    reject_forbidden_dataset_marker(str(path), context="official_file")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_bytes = path.stat().st_size
    if observed_bytes != int(expected_bytes):
        raise ValueError(
            f"official file size mismatch for {path}: "
            f"{observed_bytes} != {expected_bytes}"
        )
    observed_md5 = md5_file(path)
    if observed_md5 != str(expected_md5):
        raise ValueError(
            f"official file MD5 mismatch for {path}: "
            f"{observed_md5} != {expected_md5}"
        )
    return {
        "path": str(path),
        "bytes": observed_bytes,
        "md5": observed_md5,
        "sha256": sha256_file(path),
    }


def load_hanyang_csv(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    reject_forbidden_dataset_marker(str(path), context="source_csv")
    clip = parse_clip_name(path.name)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    if header != SOURCE_HEADER:
        raise ValueError(f"unexpected Hanyang CSV header: {path}")
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.shape != (SOURCE_FRAMES, len(SOURCE_HEADER)):
        raise ValueError(
            f"Hanyang CSV must have shape "
            f"{(SOURCE_FRAMES, len(SOURCE_HEADER))}, got {values.shape}: {path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Hanyang CSV contains non-finite values: {path}")
    expected_frames = np.arange(1, SOURCE_FRAMES + 1, dtype=np.float64)
    if not np.array_equal(values[:, 0], expected_frames):
        raise ValueError(f"Hanyang frame column must be exactly 1..150: {path}")
    positions = np.ascontiguousarray(
        values[:, 1:].reshape(SOURCE_FRAMES, len(SOURCE_JOINTS), 3)
    )
    return {
        **clip,
        "path": str(path),
        "source_sha256": sha256_file(path),
        "frames": SOURCE_FRAMES,
        "fps": SOURCE_FPS,
        "positions": positions,
        "joint_order": SOURCE_JOINTS,
    }


def _normalize_rows(values: np.ndarray, *, context: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"degenerate source vector in {context}")
    return values / norms


def anatomical_frame_rotations(positions: np.ndarray) -> np.ndarray:
    """Return source anatomical-to-source-world rotations per frame."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(SOURCE_JOINTS), 3):
        raise ValueError("Hanyang positions must have shape [frames, 19, 3]")
    index = {name: offset for offset, name in enumerate(SOURCE_JOINTS)}
    right = _normalize_rows(
        positions[:, index["RightArm"]] - positions[:, index["LeftArm"]],
        context="shoulder right axis",
    )
    up_raw = positions[:, index["Spine1"]] - positions[:, index["Hips"]]
    up = up_raw - np.sum(up_raw * right, axis=1, keepdims=True) * right
    up = _normalize_rows(up, context="torso up axis")
    forward = _normalize_rows(np.cross(right, up), context="torso front axis")
    # Re-orthogonalize after the two measured axes are normalized.
    up = _normalize_rows(np.cross(forward, right), context="orthogonal torso up")
    rotations = np.stack((forward, right, up), axis=2)
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=1e-5):
        raise ValueError("Hanyang anatomical frames must be proper rotations")
    return rotations


def robot_torso_rotations(positions: np.ndarray) -> np.ndarray:
    source = anatomical_frame_rotations(positions)
    return np.einsum("ij,fjk->fik", SOURCE_TO_ROBOT_BASIS, source)


def positions_in_robot_axes(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape [frames, joints, 3]")
    return positions @ SOURCE_TO_ROBOT_BASIS.T


def _minimal_rotation_from_z(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(1e-12, float(np.linalg.norm(direction)))
    source = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    cross = np.cross(source, direction)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, direction), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        return Rotation.from_rotvec(np.asarray((math.pi, 0.0, 0.0))).as_matrix()
    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def observed_head_rotations(positions: np.ndarray) -> np.ndarray:
    """Estimate head tilt only; axial twist/yaw is intentionally unobserved."""
    positions = np.asarray(positions, dtype=np.float64)
    index = {name: offset for offset, name in enumerate(SOURCE_JOINTS)}
    mapped = positions_in_robot_axes(positions)
    torso = robot_torso_rotations(positions)
    head_world = _normalize_rows(
        mapped[:, index["Head"]] - mapped[:, index["Neck"]],
        context="neck-to-head direction",
    )
    head_local = np.einsum("fji,fj->fi", torso, head_world)
    return np.stack(
        [_minimal_rotation_from_z(direction) for direction in head_local], axis=0
    )


def observed_head_angles(positions: np.ndarray) -> np.ndarray:
    rotations = observed_head_rotations(positions)
    xyz = Rotation.from_matrix(rotations).as_euler("XYZ")
    xyz = np.unwrap(xyz, axis=0)
    # In intrinsic XYZ, the final Z rotation does not change the observed
    # neck-to-head direction.  It is therefore removed and loss-masked.
    return np.column_stack((xyz[:, 0], xyz[:, 1], np.zeros(len(xyz))))


def observation_confidence_18d(positions: np.ndarray) -> np.ndarray:
    """Return per-frame confidence without inventing unobserved rotations."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(SOURCE_JOINTS), 3):
        raise ValueError("Hanyang positions must have shape [frames, 19, 3]")
    index = {name: offset for offset, name in enumerate(SOURCE_JOINTS)}
    confidence = np.ones((len(positions), len(JOINT_ORDER_18D)), dtype=np.float32)
    for joint in UNOBSERVED_18D_JOINTS:
        confidence[:, JOINT_ORDER_18D.index(joint)] = 0.0

    # Shoulder twist becomes singular as the elbow straightens.  The bend-plane
    # sine is zero for a straight arm and approaches one for a right-angle bend.
    for side, joint in (
        ("Left", "joint_lShoulderYaw"),
        ("Right", "joint_rShoulderYaw"),
    ):
        upper = _normalize_rows(
            positions[:, index[f"{side}ForeArm"]]
            - positions[:, index[f"{side}Arm"]],
            context=f"{side} upper arm",
        )
        fore = _normalize_rows(
            positions[:, index[f"{side}Hand"]]
            - positions[:, index[f"{side}ForeArm"]],
            context=f"{side} forearm",
        )
        bend_sine = np.linalg.norm(np.cross(upper, fore), axis=1)
        twist_confidence = np.clip((bend_sine - 0.05) / 0.20, 0.0, 1.0)
        confidence[:, JOINT_ORDER_18D.index(joint)] = twist_confidence

    head_length = np.linalg.norm(
        positions[:, index["Head"]] - positions[:, index["Neck"]], axis=1
    )
    median_length = float(np.median(head_length))
    length_confidence = np.clip(
        1.0 - np.abs(head_length / max(median_length, 1e-12) - 1.0) / 0.05,
        0.0,
        1.0,
    )
    # A center-line direction is a useful tilt proxy but not a full head pose.
    head_proxy_confidence = 0.5 * length_confidence
    for joint in ("head_roll_joint", "head_pitch_joint"):
        confidence[:, JOINT_ORDER_18D.index(joint)] = head_proxy_confidence
    return confidence


def source_geometry_quality(positions: np.ndarray) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(SOURCE_JOINTS), 3):
        raise ValueError("Hanyang positions must have shape [frames, 19, 3]")
    index = {name: offset for offset, name in enumerate(SOURCE_JOINTS)}
    upper_body_indices = [index[joint] for joint in UPPER_BODY_SOURCE_JOINTS]
    exact_zero_points = np.all(
        positions[:, upper_body_indices, :] == 0.0,
        axis=2,
    )
    exact_zero_counts_by_joint = {
        joint: int(exact_zero_points[:, offset].sum())
        for offset, joint in enumerate(UPPER_BODY_SOURCE_JOINTS)
        if np.any(exact_zero_points[:, offset])
    }
    exact_zero_source_frames = (
        np.flatnonzero(np.any(exact_zero_points, axis=1)) + 1
    ).tolist()
    segments = {
        "left_upper_arm": ("LeftArm", "LeftForeArm"),
        "left_forearm": ("LeftForeArm", "LeftHand"),
        "right_upper_arm": ("RightArm", "RightForeArm"),
        "right_forearm": ("RightForeArm", "RightHand"),
        "torso": ("Hips", "Spine1"),
        "neck_head": ("Neck", "Head"),
    }
    length_metrics: dict[str, dict[str, float]] = {}
    for name, (start, end) in segments.items():
        lengths = np.linalg.norm(
            positions[:, index[end]] - positions[:, index[start]], axis=1
        )
        mean = float(lengths.mean())
        length_metrics[name] = {
            "mean_m": mean,
            "min_m": float(lengths.min()),
            "max_m": float(lengths.max()),
            "coefficient_of_variation": float(lengths.std() / max(mean, 1e-12)),
        }
    maximum_bone_cv = max(
        metrics["coefficient_of_variation"] for metrics in length_metrics.values()
    )
    geometry_error = None
    try:
        frames = anatomical_frame_rotations(positions)
        mapped_right = (
            positions[:, index["RightArm"]] - positions[:, index["LeftArm"]]
        ) @ SOURCE_TO_ROBOT_BASIS.T
        mapped_right = _normalize_rows(mapped_right, context="mapped right axis")
        determinants = np.linalg.det(frames)
        determinant_min: float | None = float(determinants.min())
        determinant_max: float | None = float(determinants.max())
        right_axis_cosine: float | None = float(np.median(mapped_right[:, 1]))
        proper_frame_pass = bool(np.allclose(determinants, 1.0, atol=1e-5))
        left_right_axis_pass = right_axis_cosine >= 0.90
    except ValueError as error:
        geometry_error = str(error)
        determinant_min = None
        determinant_max = None
        right_axis_cosine = None
        proper_frame_pass = False
        left_right_axis_pass = False
    report = {
        "bone_lengths": length_metrics,
        "maximum_bone_length_coefficient_of_variation": maximum_bone_cv,
        "upper_body_exact_zero_point_count": int(exact_zero_points.sum()),
        "upper_body_exact_zero_frame_count": len(exact_zero_source_frames),
        "upper_body_exact_zero_source_frames": exact_zero_source_frames,
        "upper_body_exact_zero_counts_by_joint": exact_zero_counts_by_joint,
        "anatomical_frame_determinant_min": determinant_min,
        "anatomical_frame_determinant_max": determinant_max,
        "mapped_anatomical_right_robot_y_median_cosine": right_axis_cosine,
    }
    if geometry_error is not None:
        report["anatomical_frame_error"] = geometry_error
    report["quality_gate"] = {
        "no_exact_zero_upper_body_points_pass": not bool(
            exact_zero_points.any()
        ),
        "bone_length_stability_pass": maximum_bone_cv <= SOURCE_BONE_CV_LIMIT,
        "proper_anatomical_frame_pass": proper_frame_pass,
        "left_right_axis_pass": left_right_axis_pass,
    }
    report["quality_gate"]["passed"] = all(report["quality_gate"].values())
    return report


def _column_number(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    value = 0
    for character in letters:
        value = value * 26 + ord(character.upper()) - ord("A") + 1
    return value - 1


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{_XML_NS}t"))
        for item in root.findall(f"{_XML_NS}si")
    ]


def _xlsx_sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relation_id = None
    for sheet in workbook.iter(f"{_XML_NS}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relation_id = sheet.attrib.get(f"{_REL_NS}id")
            break
    if not relation_id:
        raise ValueError(f"XLSX is missing sheet {sheet_name!r}")
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relation in relationships.findall(f"{_PACKAGE_REL_NS}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"XLSX relation missing for sheet {sheet_name!r}")


def _xlsx_cell_value(cell: ElementTree.Element, shared: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{_XML_NS}t")
        )
    value = cell.find(f"{_XML_NS}v")
    text = "" if value is None or value.text is None else value.text
    if cell_type == "s" and text:
        return shared[int(text)]
    return text


def load_human_evaluations(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read the official 13-rater sheet without an openpyxl dependency."""
    path = Path(path).resolve()
    reject_forbidden_dataset_marker(str(path), context="human_evaluation_xlsx")
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive)
        sheet_path = _xlsx_sheet_path(archive, "Accuracy per motion")
        root = ElementTree.fromstring(archive.read(sheet_path))
    raw_rows: list[dict[int, str]] = []
    for row in root.iter(f"{_XML_NS}row"):
        values = {
            _column_number(cell.attrib["r"]): _xlsx_cell_value(cell, shared)
            for cell in row.findall(f"{_XML_NS}c")
        }
        raw_rows.append(values)
    if not raw_rows:
        raise ValueError("Hanyang human-evaluation sheet is empty")
    header = raw_rows[0]
    expected = {
        0: "animation_index",
        1: "True answer",
        **{index + 2: f"P{index + 1}_answer" for index in range(RATER_COUNT)},
    }
    if any(header.get(index) != label for index, label in expected.items()):
        raise ValueError("unexpected Hanyang human-evaluation header")
    records: dict[str, dict[str, Any]] = {}
    for raw in raw_rows[1:]:
        if not raw.get(0):
            continue
        clip = parse_clip_name(raw[0])
        intended = int(float(raw[1]))
        votes_list = []
        for index in range(2, 2 + RATER_COUNT):
            try:
                vote = int(float(raw.get(index, "")))
            except (TypeError, ValueError):
                continue
            votes_list.append(vote)
        votes = tuple(votes_list)
        if intended != clip["emotion_index"]:
            raise ValueError(f"evaluation intended label mismatch: {raw[0]}")
        if len(votes) < MIN_VALID_RATERS:
            raise ValueError(f"too few valid evaluation votes: {raw[0]}")
        if any(vote not in EMOTION_BY_ID for vote in votes):
            raise ValueError(f"evaluation vote out of range: {raw[0]}")
        counts = Counter(votes)
        maximum = max(counts.values())
        majority_ids = sorted(
            emotion for emotion, count in counts.items() if count == maximum
        )
        observed_raters = len(votes)
        intended_share = counts[intended] / observed_raters
        probability = {
            EMOTION_BY_ID[index]: counts[index] / observed_raters
            for index in EMOTION_BY_ID
        }
        record = {
            "clip_id": clip["clip_id"],
            "source_stem": clip["source_stem"],
            "intended_emotion_index": intended,
            "intended_emotion_id": EMOTION_BY_ID[intended],
            "rater_count": observed_raters,
            "expected_rater_count": RATER_COUNT,
            "missing_rater_count": RATER_COUNT - observed_raters,
            "votes": list(votes),
            "vote_counts": {
                EMOTION_BY_ID[index]: counts[index] for index in EMOTION_BY_ID
            },
            "soft_emotion_distribution": probability,
            "majority_emotion_ids": [
                EMOTION_BY_ID[index] for index in majority_ids
            ],
            "unique_majority_emotion_id": (
                EMOTION_BY_ID[majority_ids[0]] if len(majority_ids) == 1 else None
            ),
            "majority_share": maximum / observed_raters,
            "intended_share": intended_share,
            "rater_coverage_pass": (
                observed_raters >= HUMAN_RELIABILITY_MIN_RATERS
            ),
            "intended_high_confidence": bool(
                observed_raters >= HUMAN_RELIABILITY_MIN_RATERS
                and intended_share >= HUMAN_CONFIDENCE_THRESHOLD
            ),
            "intended_majority_agrees": intended in majority_ids,
        }
        record["sha256"] = json_hash(record)
        if clip["clip_id"] in records:
            raise ValueError(f"duplicate human evaluation: {clip['clip_id']}")
        records[clip["clip_id"]] = record
    if len(records) != SOURCE_CLIP_COUNT:
        raise ValueError(
            f"expected {SOURCE_CLIP_COUNT} human evaluations, got {len(records)}"
        )
    return records


def validate_participant_split() -> None:
    seen: dict[int, str] = {}
    for split, participants in DEFAULT_PARTICIPANT_SPLIT.items():
        for participant in participants:
            if participant in seen:
                raise ValueError(
                    f"participant {participant} crosses {seen[participant]}/{split}"
                )
            seen[participant] = split
    if set(seen) != set(range(1, PARTICIPANT_COUNT + 1)):
        raise ValueError("participant split must cover exactly participants 1..29")


validate_participant_split()


__all__ = [
    "ACTION_DIM_MASK_18D",
    "AXIS_POLICY",
    "DATASET",
    "DATASET_ID",
    "DATASET_LICENSE",
    "DATASET_REVISION",
    "DEFAULT_PARTICIPANT_SPLIT",
    "DUPLICATE_EXCLUDED_STEMS",
    "EMOTION_BY_ID",
    "EVALUATION_PROTOCOL_ANOMALY_STEMS",
    "EXACT_DUPLICATE_STEM_GROUPS",
    "HEAD_OBSERVATION_POLICY",
    "HUMAN_CONFIDENCE_THRESHOLD",
    "JOINT_ORDER_18D",
    "SOURCE_ARCHIVE_BYTES",
    "SOURCE_ARCHIVE_MD5",
    "SOURCE_FPS",
    "SOURCE_FRAMES",
    "SOURCE_HEADER",
    "SOURCE_JOINTS",
    "SOURCE_TO_ROBOT_BASIS",
    "UNOBSERVED_18D_JOINTS",
    "UPPER_BODY_SOURCE_JOINTS",
    "WRIST_OBSERVATION_POLICY",
    "anatomical_frame_rotations",
    "fixed_split_for_participant",
    "json_hash",
    "load_human_evaluations",
    "load_hanyang_csv",
    "md5_file",
    "observed_head_angles",
    "observed_head_rotations",
    "observation_confidence_18d",
    "parse_clip_name",
    "positions_in_robot_axes",
    "reject_forbidden_dataset_marker",
    "robot_torso_rotations",
    "sha256_file",
    "source_geometry_quality",
    "stable_json",
    "validate_official_file",
]
