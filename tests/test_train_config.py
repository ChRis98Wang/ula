import json

import pytest

from training.scripts.train_ula_fm import load_train_config, training_args_from_config


def test_training_args_from_yaml_config(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "runs" / "server_train"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
dataset_dir: {dataset_dir}
output_dir: {output_dir}
steps: 1234
batch_size: 16
max_episodes: 99
hidden_dim: 128
layers: 3
device: cpu
log_interval: 50
architecture: ula_mmdit_lite
semantic_tokens: 5
preview:
  every_steps: 250
  text: "紧张地解释，然后逐渐平静"
  mode: long
  duration_sec: 12
  min_segment_sec: 2
  max_segment_sec: 4
  min_segments: 3
  max_segments: 6
  max_velocity_rad_s: 2.5
  smooth_window: 7
  sampling_steps: 24
  width: 640
  height: 360
""".format(dataset_dir=dataset_dir, output_dir=output_dir),
        encoding="utf-8",
    )

    config = load_train_config(config_path)
    args = training_args_from_config(config)

    assert args.dataset_dir == str(dataset_dir)
    assert args.output_dir == str(output_dir)
    assert args.steps == 1234
    assert args.batch_size == 16
    assert args.architecture == "ula_mmdit_lite"
    assert args.semantic_tokens == 5
    assert args.preview_every_steps == 250
    assert args.preview_dir == str(output_dir / "previews")
    assert args.preview_text == "紧张地解释，然后逐渐平静"
    assert args.preview_duration_sec == 12
    assert args.preview_max_velocity_rad_s == 2.5


def test_training_config_passes_kimodo_preview_condition_ids(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
dataset_dir: /tmp/dataset
output_dir: /tmp/output
preview:
  text: "开心地向主人打招呼"
  behavior_id: Behavior.GreetingOwner01
  emotion_id: happy
""",
        encoding="utf-8",
    )

    args = training_args_from_config(load_train_config(config_path))

    assert args.preview_behavior_id == "Behavior.GreetingOwner01"
    assert args.preview_emotion_id == "happy"


def test_training_config_requires_dataset_and_output(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"steps": 10}), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_dir"):
        training_args_from_config(load_train_config(config_path))
