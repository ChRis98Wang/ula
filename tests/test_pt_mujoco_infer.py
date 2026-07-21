import hashlib
import json

import numpy as np
import pytest
import torch

from upper_body_skeleton.pt_mujoco_infer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT,
    EXPERIMENTAL_KIMODO_CHECKPOINT,
    GeneratedMotion,
    GeneratorCheckpointInfo,
    PtMotionGenerator,
    infer_motion,
    require_graphical_display,
    resolve_runtime_paths,
    run_direct_pt_session,
    validate_generator_condition_source,
    validate_generator_checkpoint,
)
from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_CONDITION_CONTRACT_VERSION,
    KIMODO_CONDITION_SCHEMA_VERSION,
    KIMODO_EMOTION_IDS,
    kimodo_condition_vectors_sha256,
)
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


_FIXTURE_VECTORS = np.zeros((len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS), 136), dtype=np.float32)


def _condition_contract(source_hash="0" * 64):
    return {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "condition_dim": 136,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "source_semantic_index_sha256": source_hash,
        "canonical_vectors_sha256": kimodo_condition_vectors_sha256(_FIXTURE_VECTORS),
    }


def _condition_bank(source_hash="0" * 64):
    return {
        **_condition_contract(source_hash),
        "vectors": torch.from_numpy(_FIXTURE_VECTORS.copy()),
        "source_semantic_index": "fixture.parquet",
    }


def _generator_checkpoint(**overrides):
    checkpoint = {
        "architecture": "ula_mmdit_lite",
        "action_dim": len(JOINT_ORDER),
        "condition_dim": 136,
        "condition_contract": _condition_contract(),
        "joint_order": JOINT_ORDER,
        "config": {"hidden_dim": 32, "layers": 2, "steps": 100},
        "model_state_dict": {"input.weight": torch.ones(2, 2)},
    }
    checkpoint.update(overrides)
    return checkpoint


def test_validate_generator_checkpoint_rejects_motion_latent_encoder():
    checkpoint = {
        "config": {"latent_dim": 128},
        "model_state_dict": {
            "backbone.0.weight": torch.ones(2, 2),
            "projection.0.weight": torch.ones(2, 2),
            "behavior_head.weight": torch.ones(2, 2),
        },
    }

    with pytest.raises(ValueError, match="only encode existing trajectories"):
        validate_generator_checkpoint(checkpoint)


def test_runtime_path_shortcuts_select_qwen_pair_and_reject_ambiguous_checkpoint():
    generator, semantic = resolve_runtime_paths(kimodo_qwen=True)
    assert generator == EXPERIMENTAL_KIMODO_CHECKPOINT
    assert semantic == DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT

    generator, semantic = resolve_runtime_paths()
    assert generator == DEFAULT_CHECKPOINT
    assert semantic == DEFAULT_SEMANTIC_ADAPTER_CHECKPOINT

    with pytest.raises(ValueError, match="already selects"):
        resolve_runtime_paths(kimodo_qwen=True, kimodo_experimental=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_runtime_paths(kimodo_qwen=True, checkpoint="custom.pt")


def test_validate_generator_checkpoint_reports_architecture_and_rejects_nonfinite_weights():
    info = validate_generator_checkpoint(_generator_checkpoint(), path="model.pt")

    assert info.path == "model.pt"
    assert info.architecture == "ula_mmdit_lite"
    assert info.action_dim == len(JOINT_ORDER)
    assert info.condition_dim == 136
    assert info.configured_steps == 100
    assert info.parameter_count == 4

    with pytest.raises(ValueError, match="non-finite"):
        validate_generator_checkpoint(
            _generator_checkpoint(model_state_dict={"input.weight": torch.tensor([float("nan")])})
        )
    with pytest.raises(ValueError, match="does not declare joint_order"):
        checkpoint = _generator_checkpoint()
        checkpoint.pop("joint_order")
        validate_generator_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="has no condition contract"):
        checkpoint = _generator_checkpoint()
        checkpoint.pop("condition_contract")
        validate_generator_checkpoint(checkpoint)


def test_pt_generator_checkpoint_is_safely_read_once_and_strictly_built(tmp_path, monkeypatch):
    path = tmp_path / "generator.pt"
    torch.save(_generator_checkpoint(), path)
    real_load = torch.load
    loads = []
    builds = []

    def counted_load(*args, **kwargs):
        loads.append((args, kwargs))
        return real_load(*args, **kwargs)

    def fake_build(checkpoint, device, *, strict):
        builds.append((checkpoint, device, strict))
        return object(), checkpoint

    monkeypatch.setattr("upper_body_skeleton.pt_mujoco_infer.torch.load", counted_load)
    monkeypatch.setattr("upper_body_skeleton.pt_mujoco_infer.model_from_checkpoint", fake_build)
    monkeypatch.setattr("upper_body_skeleton.pt_mujoco_infer.choose_device", lambda device: "cpu")

    generator = PtMotionGenerator.from_checkpoint(path, device="cpu")

    assert generator.info.condition_dim == 136
    assert len(loads) == 1
    assert loads[0][1]["weights_only"] is True
    assert len(builds) == 1
    assert builds[0][2] is True


def test_generator_condition_source_requires_exact_semantic_index_hash(tmp_path):
    semantic_path = tmp_path / "dataset" / "meta" / "semantic_index.parquet"
    semantic_path.parent.mkdir(parents=True)
    semantic_path.write_bytes(b"semantic-index")
    digest = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
    checkpoint = _generator_checkpoint(
        config={"dataset_dir": "dataset"},
        condition_contract=_condition_contract(digest),
    )
    bank = _condition_bank(digest)

    summary = validate_generator_condition_source(checkpoint, bank, repo_root=tmp_path)

    assert summary["semantic_index_sha256"] == digest
    with pytest.raises(ValueError, match="does not match"):
        validate_generator_condition_source(
            checkpoint,
            _condition_bank("0" * 64),
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="action_stats"):
        validate_generator_checkpoint(
            _generator_checkpoint(action_stats={"mean": torch.tensor([float("inf")])})
        )
    with pytest.raises(ValueError, match="joint_order"):
        validate_generator_checkpoint(_generator_checkpoint(joint_order=list(reversed(JOINT_ORDER))))
    with pytest.raises(ValueError, match="condition_dim"):
        validate_generator_checkpoint(_generator_checkpoint(condition_dim=6))
    with pytest.raises(ValueError, match="strictly positive"):
        validate_generator_checkpoint(
            _generator_checkpoint(
                action_stats={
                    "mean": torch.zeros(len(JOINT_ORDER)),
                    "std": torch.zeros(len(JOINT_ORDER)),
                }
            )
        )


def test_infer_motion_keeps_generated_trajectory_in_memory_and_postprocesses_it():
    calls = {}

    def build_condition(text, **kwargs):
        calls["condition"] = (text, kwargs)
        return np.arange(136, dtype=np.float32)

    def sample(model, **kwargs):
        calls["sample"] = (model, kwargs)
        return np.full((4, len(JOINT_ORDER)), 0.25, dtype=np.float32)

    def postprocess(values, **kwargs):
        calls["postprocess"] = kwargs
        return values * 2.0

    model = object()
    motion = infer_motion(
        model,
        _generator_checkpoint(),
        text="开心地挥手",
        behavior_id="Behavior.GreetingOwner01",
        emotion_id="happy",
        frames=4,
        fps=20,
        sampling_steps=3,
        device="cuda",
        seed=9,
        condition_builder=build_condition,
        sampler=sample,
        postprocessor=postprocess,
    )

    assert motion.trajectory.shape == (4, len(JOINT_ORDER))
    assert np.allclose(motion.raw_trajectory, 0.25)
    assert np.allclose(motion.trajectory, 0.5)
    assert calls["condition"][1]["condition_dim"] == 136
    assert calls["sample"][1]["device"] == "cuda"
    assert calls["sample"][1]["steps"] == 3
    assert motion.summary()["duration_sec"] == 0.2


def test_infer_motion_reports_semantic_adapter_resolved_labels():
    class AdapterBuilder:
        last_prediction = None

        def __call__(self, text, **kwargs):
            self.last_prediction = type(
                "Prediction",
                (),
                {
                    "behavior_id": "Behavior.GreetingOwner01",
                    "emotion_id": "happy",
                    "behavior_confidence": 0.8,
                    "emotion_confidence": 0.9,
                },
            )()
            return np.zeros(136, dtype=np.float32)

    motion = infer_motion(
        object(),
        _generator_checkpoint(),
        text="开心地挥手",
        frames=4,
        condition_builder=AdapterBuilder(),
        sampler=lambda model, **kwargs: np.zeros((4, len(JOINT_ORDER)), dtype=np.float32),
    )

    assert motion.behavior_id == "Behavior.GreetingOwner01"
    assert motion.emotion_id == "happy"
    assert motion.summary()["behavior_confidence"] == pytest.approx(0.8)
    assert motion.summary()["emotion_confidence"] == pytest.approx(0.9)


def test_infer_motion_rejects_structured_labels_for_legacy_conditioning():
    checkpoint = _generator_checkpoint(architecture="ula_fm_legacy", condition_dim=92)

    with pytest.raises(ValueError, match="do not support structured"):
        infer_motion(
            object(),
            checkpoint,
            text="wave",
            behavior_id="Behavior.GreetingOwner01",
            frames=4,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fps": float("nan")}, "fps must be finite"),
        ({"fps": float("inf")}, "fps must be finite"),
        ({"max_velocity_rad_s": float("inf")}, "max_velocity_rad_s must be finite"),
        ({"smooth_window": 0}, "smooth_window must be at least 1"),
    ],
)
def test_infer_motion_rejects_nonfinite_or_unsafe_postprocess_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        infer_motion(
            object(),
            _generator_checkpoint(),
            text="test",
            frames=4,
            condition_builder=lambda *args, **kwargs: np.zeros(136, dtype=np.float32),
            sampler=lambda model, **kwargs: np.zeros((4, len(JOINT_ORDER)), dtype=np.float32),
            **kwargs,
        )


def test_direct_session_sends_in_memory_array_to_reusable_mujoco_player():
    printed = []

    class FakeGenerator:
        info = GeneratorCheckpointInfo(
            path="generator.pt",
            architecture="ula_fm_legacy",
            action_dim=15,
            condition_dim=92,
            hidden_dim=32,
            layers=2,
            configured_steps=10,
            checkpoint_step=10,
            episodes_loaded=4,
            parameter_count=100,
            has_action_stats=False,
        )
        device = "cpu"

        def __init__(self):
            self.prompts = []

        def infer(self, text, **kwargs):
            self.prompts.append((text, kwargs))
            values = np.zeros((3, len(JOINT_ORDER)), dtype=np.float32)
            return GeneratedMotion(text, None, None, values, values.copy(), 30.0, 2, kwargs["seed"])

    class RunningViewer:
        def is_running(self):
            return True

    class FakePlayer:
        def __init__(self):
            self.viewer = RunningViewer()
            self.played = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def play_trajectory(self, trajectory, **kwargs):
            self.played.append((trajectory.copy(), kwargs))
            return {"frames_played": len(trajectory), "loops_completed": kwargs["loops"]}

    generator = FakeGenerator()
    player = FakePlayer()
    semantic_builder = object()
    result = run_direct_pt_session(
        generator,
        player,
        ["第一个动作", "", "第二个动作", ":q", "不会运行"],
        sampling_steps=2,
        loops=1,
        condition_builder=semantic_builder,
        writer=printed.append,
    )

    assert result["runs"] == 2
    assert [item[0] for item in generator.prompts] == ["第一个动作", "第二个动作"]
    assert [item[1]["seed"] for item in generator.prompts] == [7, 8]
    assert all(item[1]["condition_builder"] is semantic_builder for item in generator.prompts)
    assert len(player.played) == 2
    assert all(call[0].shape == (3, len(JOINT_ORDER)) for call in player.played)
    assert json.loads(printed[0])["viewer"]["frames_played"] == 3


def test_graphical_display_check_explains_remote_requirement(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("upper_body_skeleton.pt_mujoco_infer.sys.platform", "linux")

    with pytest.raises(RuntimeError, match="ssh -Y"):
        require_graphical_display()

    monkeypatch.setenv("DISPLAY", ":10")
    require_graphical_display()
