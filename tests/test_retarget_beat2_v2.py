from pathlib import Path

import numpy as np
import pytest

from tools.gmr_v2.retarget_beat2_v2 import (
    BEAT2_AXIS_POLICY,
    BEAT2_DATASET_ID,
    BEAT2_SOURCE_REVISION,
    build_beat2_provenance,
    decode_beat2_smplx,
    output_stem,
    parse_args,
)
from upper_body_skeleton.retarget_v2_18d import CONTRACT_VERSION


def write_beat2_fixture(path: Path, *, frames=8, fps=30):
    poses = np.arange(frames * 165, dtype=np.float32).reshape(frames, 165) / 1000.0
    trans = np.arange(frames * 3, dtype=np.float64).reshape(frames, 3) / 100.0
    betas = np.linspace(-1.0, 1.0, 300, dtype=np.float32)
    expressions = np.full((frames, 100), np.nan)
    np.savez(
        path,
        poses=poses,
        trans=trans,
        betas=betas,
        expressions=expressions,
        model=np.asarray("smplx2020"),
        gender=np.asarray("neutral"),
        mocap_frame_rate=np.asarray(fps, dtype=np.int32),
    )
    return poses, trans, betas


def test_decode_beat2_uses_exact_global_body_trans_and_fixed_betas(tmp_path):
    path = tmp_path / "speaker_clip.npz"
    poses, trans, betas = write_beat2_fixture(path)

    decoded = decode_beat2_smplx(path, start_frame=2, end_frame=6)

    assert np.array_equal(decoded["root_orient"], poses[2:6, 0:3])
    assert np.array_equal(decoded["pose_body"], poses[2:6, 3:66])
    assert np.array_equal(decoded["trans"], trans[2:6].astype(np.float32))
    assert decoded["betas"].shape == (1, 10)
    assert np.array_equal(decoded["betas"][0], betas[:10])
    assert decoded["frame_count"] == 4
    assert decoded["source_total_frames"] == 8
    assert decoded["source_start_frame"] == 2
    assert decoded["source_end_frame"] == 6
    assert decoded["mocap_frame_rate"] == 30.0


def test_decode_beat2_ignores_hand_face_and_expression_nonfinite_values(tmp_path):
    path = tmp_path / "ignored_channels.npz"
    poses, trans, betas = write_beat2_fixture(path)
    poses[:, 66:] = np.nan
    np.savez(
        path,
        poses=poses,
        trans=trans,
        betas=betas,
        expressions=np.full((len(poses), 100), np.nan),
        model=np.asarray("smplx2020"),
        gender=np.asarray("neutral"),
        mocap_frame_rate=np.asarray(30),
    )

    decoded = decode_beat2_smplx(path)

    assert np.isfinite(decoded["root_orient"]).all()
    assert np.isfinite(decoded["pose_body"]).all()


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 4), (0, 9), (4, 4), (5, 4), (0, 2)],
)
def test_decode_beat2_rejects_invalid_half_open_windows(tmp_path, start, end):
    path = tmp_path / "window.npz"
    write_beat2_fixture(path)

    with pytest.raises(ValueError, match="window|at least three"):
        decode_beat2_smplx(path, start_frame=start, end_frame=end)


@pytest.mark.parametrize("column", [0, 65])
def test_decode_beat2_rejects_nonfinite_mapped_pose(tmp_path, column):
    path = tmp_path / "bad_pose.npz"
    poses, trans, betas = write_beat2_fixture(path)
    poses[3, column] = np.nan
    np.savez(
        path,
        poses=poses,
        trans=trans,
        betas=betas,
        mocap_frame_rate=np.asarray(30),
    )

    with pytest.raises(ValueError, match="global/body pose"):
        decode_beat2_smplx(path)


def test_decode_beat2_requires_clip_level_beta_vector(tmp_path):
    path = tmp_path / "bad_betas.npz"
    poses, trans, _ = write_beat2_fixture(path)
    np.savez(
        path,
        poses=poses,
        trans=trans,
        betas=np.zeros((len(poses), 10), dtype=np.float32),
        mocap_frame_rate=np.asarray(30),
    )

    with pytest.raises(ValueError, match="clip-level BEAT2 betas"):
        decode_beat2_smplx(path)


def test_beat2_provenance_is_explicit_about_window_and_ignored_pose(tmp_path):
    path = tmp_path / "speaker_clip.npz"
    write_beat2_fixture(path)
    decoded = decode_beat2_smplx(path, start_frame=1, end_frame=7)

    provenance = build_beat2_provenance(path, decoded, "a" * 64)

    assert provenance["source_dataset_id"] == BEAT2_DATASET_ID
    assert provenance["source_revision"] == BEAT2_SOURCE_REVISION
    assert provenance["source_window_start_frame"] == 1
    assert provenance["source_window_end_frame_exclusive"] == 7
    assert provenance["source_window_convention"] == "zero_based_half_open_[start,end)"
    assert provenance["pose_decode"] == {
        "global_orient": "poses[:, 0:3]",
        "body_pose": "poses[:, 3:66]",
        "ignored_pose": "poses[:, 66:165]",
    }
    assert provenance["betas_policy"] == (
        "first_10_clip_level_coefficients_fixed_across_all_frames"
    )
    assert len(provenance["selected_betas_sha256"]) == 64
    assert output_stem(path, decoded) == "speaker_clip_f000001-000007"


def test_cli_is_18d_only_and_end_frame_is_exclusive(tmp_path):
    args = parse_args(
        [
            "--beat2",
            str(tmp_path / "source.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--start-frame",
            "10",
            "--end-frame",
            "40",
        ]
    )

    assert args.output_contract == CONTRACT_VERSION
    assert args.start_frame == 10
    assert args.end_frame == 40
    assert BEAT2_AXIS_POLICY.startswith("beat2_smplx_")
