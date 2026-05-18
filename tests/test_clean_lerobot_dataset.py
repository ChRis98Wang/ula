import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from upper_body_skeleton.clean_lerobot_dataset import clean_episode_trajectory, clean_lerobot_dataset, trajectory_quality
from upper_body_skeleton.lerobot_export import data_schema


def test_clean_episode_trajectory_removes_single_frame_spike_without_flattening_motion():
    frames = 60
    trajectory = np.zeros((frames, 2), dtype=np.float32)
    trajectory[:, 0] = np.linspace(0.0, 1.18, frames, dtype=np.float32)
    trajectory[:, 1] = 0.15 * np.sin(np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32))
    reference_without_spike = trajectory.copy()
    trajectory[30, 0] = 3.8

    before = trajectory_quality(trajectory, fps=30.0)
    valid_reference = trajectory_quality(reference_without_spike, fps=30.0)
    cleaned, report = clean_episode_trajectory(
        trajectory,
        fps=30.0,
        max_velocity_rad_s=6.0,
        spike_delta_rad=0.35,
        spike_neighbor_ratio=2.5,
        smoothing_window=3,
        min_energy_ratio=0.85,
    )
    after = trajectory_quality(cleaned, fps=30.0)

    assert before["max_delta_rad"] > 2.0
    assert after["max_delta_rad"] <= 0.35 + 1e-6
    assert after["max_velocity_rad_s"] <= 6.0 + 1e-5
    assert report["spike_replacements"] >= 1
    assert report["valid_energy_ratio"] >= 0.85
    assert report["raw_energy_ratio"] < report["valid_energy_ratio"]
    assert after["mean_velocity_rad_s"] >= valid_reference["mean_velocity_rad_s"] * 0.85


def test_clean_episode_trajectory_preserves_deliberate_fast_ramp_energy():
    frames = 80
    trajectory = np.zeros((frames, 1), dtype=np.float32)
    trajectory[20:50, 0] = np.linspace(0.0, 1.2, 30, dtype=np.float32)
    trajectory[50:, 0] = 1.2

    cleaned, report = clean_episode_trajectory(
        trajectory,
        fps=30.0,
        max_velocity_rad_s=6.0,
        spike_delta_rad=0.35,
        spike_neighbor_ratio=2.5,
        smoothing_window=3,
        min_energy_ratio=0.9,
    )

    assert np.max(np.abs(cleaned - trajectory)) < 0.08
    assert report["valid_energy_ratio"] >= 0.9
    assert report["raw_energy_ratio"] >= 0.9
    assert report["velocity_limited_steps"] == 0


def test_clean_episode_trajectory_respects_valid_energy_floor_after_velocity_limit():
    trajectory = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-0.08319872617721558, 0.06003609672188759, 0.07524517923593521],
            [-0.3892815411090851, 0.10586173832416534, 0.08547241240739822],
            [-0.41458094120025635, 0.10451764613389969, 0.01722889579832554],
            [-0.3442291021347046, 0.1667409986257553, 0.022511351853609085],
            [-0.404049813747406, 0.2041417509317398, -0.04623204469680786],
            [-0.3745497465133667, 0.12743113934993744, 0.02404397912323475],
        ],
        dtype=np.float32,
    )

    cleaned, report = clean_episode_trajectory(
        trajectory,
        fps=30.0,
        max_velocity_rad_s=6.0,
        spike_delta_rad=0.35,
        spike_neighbor_ratio=2.5,
        smoothing_window=5,
        min_energy_ratio=0.9,
    )

    assert report["valid_energy_ratio"] >= 0.9 - 1e-6
    assert trajectory_quality(cleaned, fps=30.0)["max_velocity_rad_s"] <= 6.0 + 1e-5


def test_clean_lerobot_dataset_writes_clean_copy_with_consistent_actions(tmp_path):
    input_dir = tmp_path / "lerobot"
    data_dir = input_dir / "data" / "chunk-000"
    meta_dir = input_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    (input_dir / "README.md").write_text("source\n", encoding="utf-8")
    (meta_dir / "info.json").write_text("{}", encoding="utf-8")

    rows = []
    values = [0.0, 0.1, 3.2, 0.2, 0.3]
    for frame_index, value in enumerate(values):
        state = [float(value)] + [0.0] * 14
        action = state if frame_index == len(values) - 1 else [float(values[frame_index + 1])] + [0.0] * 14
        rows.append(
            {
                "index": frame_index,
                "episode_index": 0,
                "frame_index": frame_index,
                "timestamp": frame_index / 30.0,
                "task_index": 0,
                "observation.state": state,
                "action": action,
                "next.done": frame_index == len(values) - 1,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=data_schema()), data_dir / "file-000.parquet")

    output_dir = tmp_path / "lerobot_clean"
    summary = clean_lerobot_dataset(
        input_dir,
        output_dir,
        max_velocity_rad_s=6.0,
        spike_delta_rad=0.35,
        smoothing_window=1,
    )

    cleaned_rows = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet").to_pylist()
    cleaned_values = [row["observation.state"][0] for row in cleaned_rows]
    cleaned_actions = [row["action"][0] for row in cleaned_rows]
    assert max(abs(np.diff(cleaned_values))) <= 0.2 + 1e-6
    assert cleaned_actions[:-1] == cleaned_values[1:]
    assert cleaned_actions[-1] == cleaned_values[-1]
    assert summary["spike_replacements"] >= 1
    assert (output_dir / "meta" / "info.json").exists()
    assert (output_dir / "cleaning_summary.json").exists()


def test_clean_lerobot_dataset_preserves_actions_when_episode_spans_parquet_files(tmp_path):
    input_dir = tmp_path / "lerobot"
    data_dir = input_dir / "data" / "chunk-000"
    meta_dir = input_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    (input_dir / "README.md").write_text("source\n", encoding="utf-8")
    (meta_dir / "info.json").write_text('{"chunks_size":3}', encoding="utf-8")

    values = [0.0, 0.1, 3.2, 0.2, 0.3]
    rows = []
    for frame_index, value in enumerate(values):
        state = [float(value)] + [0.0] * 14
        action = state if frame_index == len(values) - 1 else [float(values[frame_index + 1])] + [0.0] * 14
        rows.append(
            {
                "index": frame_index,
                "episode_index": 7,
                "frame_index": frame_index,
                "timestamp": frame_index / 30.0,
                "task_index": 0,
                "observation.state": state,
                "action": action,
                "next.done": frame_index == len(values) - 1,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows[:3], schema=data_schema()), data_dir / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[3:], schema=data_schema()), data_dir / "file-001.parquet")

    output_dir = tmp_path / "lerobot_clean"
    summary = clean_lerobot_dataset(
        input_dir,
        output_dir,
        max_velocity_rad_s=6.0,
        spike_delta_rad=0.35,
        smoothing_window=1,
    )

    cleaned_rows = []
    for path in sorted((output_dir / "data" / "chunk-000").glob("file-*.parquet")):
        cleaned_rows.extend(pq.read_table(path).to_pylist())
    cleaned_values = [row["observation.state"][0] for row in cleaned_rows]
    cleaned_actions = [row["action"][0] for row in cleaned_rows]
    assert summary["episodes"] == 1
    assert cleaned_actions[:-1] == cleaned_values[1:]
    assert cleaned_actions[-1] == cleaned_values[-1]
