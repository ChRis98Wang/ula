from pathlib import Path

from upper_body_skeleton.batch_retarget import (
    discover_npz_files,
    output_paths_for_npz,
    resolve_max_frames,
    resolve_retarget_mode,
    should_write_progress_manifest,
    should_reuse_skeleton,
    select_shard,
)


def test_discover_npz_files_returns_stable_sorted_order(tmp_path):
    extracted = tmp_path / "extracted"
    (extracted / "b").mkdir(parents=True)
    (extracted / "a").mkdir(parents=True)
    (extracted / "b" / "sample_b.npz").write_bytes(b"")
    (extracted / "a" / "sample_a.npz").write_bytes(b"")
    (extracted / "a" / "ignore.mp4").write_bytes(b"")

    discovered = discover_npz_files(extracted)

    assert [path.name for path in discovered] == ["sample_a.npz", "sample_b.npz"]


def test_output_paths_preserve_relative_parent_to_avoid_collisions(tmp_path):
    extracted = tmp_path / "extracted"
    output_root = tmp_path / "batch"
    npz_path = extracted / "session_a" / "clip_001" / "same_name.npz"

    paths = output_paths_for_npz(npz_path, extracted, output_root)

    assert paths.work_dir == output_root / "session_a" / "clip_001" / "same_name"
    assert paths.skeleton_json.name == "same_name.keypoint_upper_body_skeleton_smoothed.json"
    assert paths.joint_csv.name == "same_name.v2_upper_body_joints.csv"
    assert paths.monitor_json.name == "same_name.v2_monitor_report.json"
    assert Path(paths.manifest_key) == Path("session_a") / "clip_001" / "same_name"


def test_resolve_max_frames_uses_none_for_full_video_mode():
    assert resolve_max_frames(max_frames=180, full_video=True) is None
    assert resolve_max_frames(max_frames=240, full_video=False) == 240


def test_resolve_retarget_mode_accepts_formula_and_ik():
    assert resolve_retarget_mode("formula") == "formula"
    assert resolve_retarget_mode("ik") == "ik"


def test_select_shard_partitions_files_without_overlap():
    files = [Path(f"sample_{index}.npz") for index in range(10)]

    shards = [select_shard(files, shard_index=index, shard_count=3) for index in range(3)]
    flattened = [path for shard in shards for path in shard]

    assert sorted(flattened) == files
    assert len(set(flattened)) == len(files)
    assert shards[0] == [files[0], files[3], files[6], files[9]]


def test_should_reuse_skeleton_only_when_present_and_not_overwriting(tmp_path):
    skeleton = tmp_path / "sample.keypoint_upper_body_skeleton_smoothed.json"

    assert not should_reuse_skeleton(skeleton, overwrite=False)
    skeleton.write_text("{}", encoding="utf-8")
    assert should_reuse_skeleton(skeleton, overwrite=False)
    assert not should_reuse_skeleton(skeleton, overwrite=True)


def test_should_write_progress_manifest_for_first_interval_and_final_rows():
    assert should_write_progress_manifest(index=1, total=10, interval=25)
    assert should_write_progress_manifest(index=25, total=100, interval=25)
    assert should_write_progress_manifest(index=10, total=10, interval=25)
    assert not should_write_progress_manifest(index=2, total=10, interval=25)
    assert not should_write_progress_manifest(index=2, total=10, interval=0)
