from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.gmr_v2.interact_bvh_adapter import (
    CANONICAL_BODY_ORDER,
    GMR_FRONT_REFLECTION,
    GMR_IN_PLANE_BASIS_ROTATION,
    GMR_SOURCE_TO_ROBOT_BASIS,
    INTERACT_BVH_FORWARD_AXIS,
    INTERACT_BVH_RIGHT_AXIS,
    INTERACT_BVH_UP_AXIS,
    INTERACT_NATIVE_AXIS_POLICY,
    INTERACT_NATIVE_TO_ROBOT_BASIS,
    SOURCE_TO_CANONICAL,
    adapt_interact_bvh,
    anatomical_frames_from_landmarks,
    adapter_smoke_report,
    canonicalize_gmr_global_data,
    head_rotations_in_anatomical_parent_frame,
    is_30hz,
    load_interact_bvh_native_v2,
    local_rotation_matrices,
    parse_bvh,
    read_bvh_header,
)


JOINT_TREE = (
    ("Spine", "Hips"),
    ("Spine1", "Spine"),
    ("Spine2", "Spine1"),
    ("Spine3", "Spine2"),
    ("Neck", "Spine3"),
    ("Neck1", "Neck"),
    ("Head", "Neck1"),
    ("LeftArm", "Spine3"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("RightArm", "Spine3"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
)


def _children(parent: str):
    return [name for name, direct_parent in JOINT_TREE if direct_parent == parent]


def _joint_block(name: str, depth: int = 0) -> list[str]:
    indent = "\t" * depth
    declaration = "ROOT" if name == "Hips" else "JOINT"
    offset = {
        "Hips": (0, 0, 0),
        "Spine": (0, 10, 0),
        "Spine1": (0, 10, 0),
        "Spine2": (0, 10, 0),
        "Spine3": (0, 10, 0),
        "Neck": (0, 10, 0),
        "Neck1": (0, 5, 0),
        "Head": (0, 5, 0),
        "LeftArm": (0, 0, -15),
        "LeftForeArm": (0, 0, -25),
        "LeftHand": (0, 0, -20),
        "RightArm": (0, 0, 15),
        "RightForeArm": (0, 0, 25),
        "RightHand": (0, 0, 20),
    }[name]
    rows = [f"{indent}{declaration} {name}", f"{indent}{{"]
    rows.append(f"{indent}\tOFFSET {offset[0]} {offset[1]} {offset[2]}")
    if name == "Hips":
        rows.append(
            f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
        )
    else:
        rows.append(f"{indent}\tCHANNELS 3 Zrotation Yrotation Xrotation")
    for child in _children(name):
        rows.extend(_joint_block(child, depth + 1))
    rows.append(f"{indent}}}")
    return rows


def write_fixture(path: Path, frame_count: int = 8, *, moving: bool = True) -> None:
    hierarchy = ["HIERARCHY", *_joint_block("Hips")]
    total_channels = 6 + 3 * len(JOINT_TREE)
    frames = []
    for frame in range(frame_count):
        values = np.zeros(total_channels)
        if moving:
            values[3] = frame * 2.0
            # First non-root joint rotation starts at channel six.
            values[6] = frame * 1.5
            values[-3] = frame * 3.0
        frames.append(" ".join(f"{value:.6f}" for value in values))
    payload = "\n".join(
        [
            *hierarchy,
            "MOTION",
            f"Frames: {frame_count}",
            "Frame Time: 0.033333",
            *frames,
        ]
    )
    path.write_text(payload + "\n", encoding="utf-8")


def test_header_and_structure_are_30hz_and_complete(tmp_path):
    path = tmp_path / "clip.bvh"
    write_fixture(path)
    frame_count, frame_time = read_bvh_header(path)
    assert frame_count == 8
    assert is_30hz(frame_time)
    structure = parse_bvh(path)
    assert structure.motion.shape == (8, 45)
    assert structure.names[0] == "Hips"
    assert structure.parents[structure.names.index("Head")] == structure.names.index(
        "Neck1"
    )


def test_adapter_maps_arms_and_preserves_three_joint_head_chain(tmp_path):
    path = tmp_path / "clip.bvh"
    write_fixture(path)
    adapted = adapt_interact_bvh(path)
    assert adapted["source_to_canonical"] == SOURCE_TO_CANONICAL
    assert adapted["head_chain"] == ["Spine3", "Neck", "Neck1", "Head"]
    assert adapted["head_relative_rotations"].shape == (8, 3, 3)
    assert adapted["head_native_unaligned_3dof"].shape == (8, 3)
    assert adapted["canonical_positions_stacked"].shape == (8, 7, 3)
    assert adapted["canonical_quaternions_stacked_wxyz"].shape == (8, 7, 4)
    for canonical in SOURCE_TO_CANONICAL.values():
        assert adapted["canonical_positions"][canonical].shape == (8, 3)
        assert adapted["canonical_quaternions_wxyz"][canonical].shape == (8, 4)


def test_adapter_smoke_never_self_admits_unverified_axes(tmp_path):
    path = tmp_path / "clip.bvh"
    write_fixture(path)
    report = adapter_smoke_report(path)
    assert report["parser_mapping_smoke_passed"] is True
    assert report["finger_joints_used"] is False
    assert report["face_channels_used"] is False
    assert report["audio_used"] is False
    assert report["accepted_for_18d_retarget"] is False
    assert report["axis_visual_qc_status"].startswith("pending")


def test_missing_required_joint_fails_closed(tmp_path):
    path = tmp_path / "clip.bvh"
    write_fixture(path)
    payload = path.read_text(encoding="utf-8").replace("JOINT Neck1", "JOINT Other")
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="missing adapter joints"):
        adapt_interact_bvh(path)


def test_gmr_adapter_emits_existing_stacked_source_contract():
    names = (
        "Hips",
        "Spine3",
        "Neck",
        "Neck1",
        "Head",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
    )
    positions = np.zeros((3, len(names), 3), dtype=float)
    name_to_index = {name: index for index, name in enumerate(names)}
    for frame in range(3):
        hips = np.array([0.1 * frame, 0.0, 0.0])
        chest = hips + np.array([0.0, 0.0, 1.0])
        positions[frame, name_to_index["Hips"]] = hips
        positions[frame, name_to_index["Spine3"]] = chest
        positions[frame, name_to_index["LeftArm"]] = chest + [0.0, -0.4, 0.0]
        positions[frame, name_to_index["RightArm"]] = chest + [0.0, 0.4, 0.0]
        positions[frame, name_to_index["Neck"]] = chest + [0.0, 0.0, 0.1]
        positions[frame, name_to_index["Neck1"]] = chest + [0.0, 0.0, 0.2]
        positions[frame, name_to_index["Head"]] = chest + [0.0, 0.0, 0.3]
        positions[frame, name_to_index["LeftForeArm"]] = chest + [0.0, -0.7, -0.1]
        positions[frame, name_to_index["LeftHand"]] = chest + [0.0, -0.9, -0.2]
        positions[frame, name_to_index["RightForeArm"]] = chest + [0.0, 0.7, -0.1]
        positions[frame, name_to_index["RightHand"]] = chest + [0.0, 0.9, -0.2]
    quaternions = np.zeros((3, len(names), 4), dtype=float)
    quaternions[..., 0] = 1.0
    result = canonicalize_gmr_global_data(
        names, positions, quaternions, start_frame=1, end_frame=3
    )
    assert result["canonical_body_order"] == list(CANONICAL_BODY_ORDER)
    assert result["canonical_positions_stacked"].shape == (2, 7, 3)
    assert result["canonical_quaternions_stacked_wxyz"].shape == (2, 7, 4)
    assert result["head_relative_rotations"].shape == (2, 3, 3)
    assert result["review_joint_positions"].shape == (2, 9, 3)
    spine3 = names.index("Spine3")
    np.testing.assert_allclose(
        result["canonical_positions_stacked"][:, 0],
        positions[1:3, spine3] @ GMR_SOURCE_TO_ROBOT_BASIS.T,
    )


def test_gmr_source_to_robot_basis_maps_anatomical_axes_explicitly():
    np.testing.assert_allclose(
        GMR_SOURCE_TO_ROBOT_BASIS.T @ GMR_SOURCE_TO_ROBOT_BASIS,
        np.eye(3),
        atol=1e-12,
    )
    assert np.linalg.det(GMR_FRONT_REFLECTION) == pytest.approx(-1.0)
    assert np.linalg.det(GMR_IN_PLANE_BASIS_ROTATION) == pytest.approx(1.0)
    assert np.linalg.det(GMR_SOURCE_TO_ROBOT_BASIS) == pytest.approx(-1.0)
    np.testing.assert_allclose(
        GMR_SOURCE_TO_ROBOT_BASIS @ np.array([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        GMR_SOURCE_TO_ROBOT_BASIS @ np.array([0.0, 1.0, 0.0]),
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        GMR_SOURCE_TO_ROBOT_BASIS @ np.array([0.0, 0.0, 1.0]),
        [0.0, 0.0, 1.0],
    )


def test_native_v2_loader_maps_declared_bvh_basis_with_explicit_handedness(tmp_path):
    path = tmp_path / "clip.bvh"
    write_fixture(path, moving=False)
    result = load_interact_bvh_native_v2(path, start_frame=1, end_frame=7)

    assert result["axis_policy"] == INTERACT_NATIVE_AXIS_POLICY
    assert result["legacy_gmr_euler_component_reorder_used"] is False
    assert result["bvh_rotation_composition"] == "declared_channel_order_intrinsic"
    assert result["axis_alignment_determinant"] == pytest.approx(-1.0)
    np.testing.assert_allclose(
        INTERACT_NATIVE_TO_ROBOT_BASIS @ INTERACT_BVH_FORWARD_AXIS,
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        INTERACT_NATIVE_TO_ROBOT_BASIS @ INTERACT_BVH_RIGHT_AXIS,
        [0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        INTERACT_NATIVE_TO_ROBOT_BASIS @ INTERACT_BVH_UP_AXIS,
        [0.0, 0.0, 1.0],
    )

    positions = result["canonical_positions_stacked"]
    index = {name: offset for offset, name in enumerate(CANONICAL_BODY_ORDER)}
    chest = positions[0, index["Chest4"]]
    np.testing.assert_allclose(
        positions[0, index["LeftShoulder"]] - chest,
        [0.0, -0.15, 0.0],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        positions[0, index["RightShoulder"]] - chest,
        [0.0, 0.15, 0.0],
        atol=1e-9,
    )
    assert result["review_joint_rotations"].shape == (6, 9, 3, 3)


def test_head_orientation_uses_anatomical_torso_not_spine3_sensor_frame():
    torso = Rotation.from_euler(
        "XYZ", [[10.0, -15.0, 25.0], [-8.0, 12.0, 31.0]], degrees=True
    ).as_matrix()
    false_spine_roll = Rotation.from_euler(
        "X", [30.0, -24.0], degrees=True
    ).as_matrix()
    spine3 = torso @ false_spine_roll
    head = torso.copy()

    actual = head_rotations_in_anatomical_parent_frame(torso, head)
    np.testing.assert_allclose(
        actual, np.broadcast_to(np.eye(3), actual.shape), atol=1e-12
    )
    old_spine3_relative = np.swapaxes(spine3, 1, 2) @ head
    assert np.rad2deg(Rotation.from_matrix(old_spine3_relative).magnitude()).min() > 20.0


def test_native_bvh_fk_composes_rotations_in_declared_channel_order(tmp_path):
    path = tmp_path / "clip.bvh"
    write_fixture(path, frame_count=3, moving=False)
    structure = parse_bvh(path)
    joint = structure.names.index("Head")
    start = int(structure.channel_starts[joint])
    assert structure.channel_names[joint] == (
        "Zrotation",
        "Yrotation",
        "Xrotation",
    )
    structure.motion[1, start : start + 3] = [31.0, -17.0, 23.0]
    actual = local_rotation_matrices(structure, ["Head"])["Head"][1]
    expected = Rotation.from_euler("ZYX", [31.0, -17.0, 23.0], degrees=True).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_anatomical_frame_uses_right_up_forward_without_swapping_sides():
    hips = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.0]])
    chest = hips + np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    left = chest + np.array([[0.0, -0.4, 0.0], [0.0, -0.4, 0.0]])
    right = chest + np.array([[0.0, 0.4, 0.0], [0.0, 0.4, 0.0]])
    frames = anatomical_frames_from_landmarks(hips, chest, left, right)
    np.testing.assert_allclose(frames[:, :, 0], [[1.0, 0.0, 0.0]] * 2)
    np.testing.assert_allclose(frames[:, :, 1], [[0.0, 1.0, 0.0]] * 2)
    np.testing.assert_allclose(frames[:, :, 2], [[0.0, 0.0, 1.0]] * 2)
    np.testing.assert_allclose(np.linalg.det(frames), np.ones(2))
