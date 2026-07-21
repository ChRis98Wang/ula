import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from training.scripts.train_semantic_adapter import training_args_from_config
from upper_body_skeleton.kimodo_semantics import (
    KIMODO_BEHAVIOR_IDS,
    KIMODO_CONDITION_CONTRACT_VERSION,
    KIMODO_CONDITION_SCHEMA_VERSION,
    KIMODO_EMOTION_IDS,
    kimodo_condition_vectors_sha256,
)
from upper_body_skeleton.semantic_adapter import (
    AdapterConditionBuilder,
    FrozenQwenSemanticAdapter,
    SemanticAdapterNetwork,
    SemanticPrediction,
    SemanticPromptRecord,
    build_deployment_semantic_records,
    build_multilingual_semantic_records,
    evaluate_semantic_adapter,
    last_token_pool,
    latin_square_semantic_split,
    load_semantic_adapter,
    load_semantic_paraphrase_config,
    load_semantic_prompt_catalog,
    semantic_adapter_checkpoint_payload,
    semantic_adapter_loss,
    train_semantic_adapter_head,
    validate_condition_bank,
    validate_semantic_adapter_checkpoint,
)


def _records():
    return [
        SemanticPromptRecord(behavior, emotion, f"instruction {behavior} {emotion}")
        for behavior in KIMODO_BEHAVIOR_IDS
        for emotion in KIMODO_EMOTION_IDS
    ]


def _write_catalog(path):
    lines = ["behavior_id,emotion_id,prompt"]
    lines.extend(f'{row.behavior_id},{row.emotion_id},"{row.text}"' for row in _records())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paraphrases(path):
    payload = {
        "schema_version": 1,
        "behaviors": {
            behavior: {
                "train_phrases": [f"训练动作甲{index}", f"训练动作乙{index}"],
                "validation_phrase": f"验证动作{index}",
                "test_phrase": f"测试动作{index}",
            }
            for index, behavior in enumerate(KIMODO_BEHAVIOR_IDS)
        },
        "emotions": {
            emotion: {
                "train_prefixes": [f"训练情绪甲{index}", f"训练情绪乙{index}"],
                "validation_prefix": f"验证情绪{index}",
                "test_prefix": f"测试情绪{index}",
            }
            for index, emotion in enumerate(KIMODO_EMOTION_IDS)
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FakeTextEncoder:
    embedding_dim = 33

    def encode(self, texts, *, batch_size=16):
        values = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            values[row, sum(text.encode("utf-8")) % self.embedding_dim] = 1.0
        return values


class FixedSemanticAdapter:
    def predict_one(self, text):
        return SemanticPrediction(text, "Behavior.GreetingOwner01", "happy", 0.8, 0.9)


def _condition_bank():
    vectors = torch.zeros(len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS), 136)
    return {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "condition_dim": 136,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "vectors": vectors,
        "source_semantic_index": "fixture.parquet",
        "source_semantic_index_sha256": "0" * 64,
        "canonical_vectors_sha256": kimodo_condition_vectors_sha256(vectors.numpy()),
    }


def test_catalog_loader_requires_complete_strict_label_grid(tmp_path):
    path = tmp_path / "catalog.csv"
    _write_catalog(path)

    records = load_semantic_prompt_catalog(path)

    assert len(records) == 162
    assert len({row.key for row in records}) == 162
    path.write_text("behavior_id,emotion_id,prompt\nBehavior.Unknown,happy,bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown Kimodo behavior_id"):
        load_semantic_prompt_catalog(path)


def test_latin_square_split_holds_out_one_emotion_per_behavior_and_rotates_all_pairs():
    records = _records()
    test_counts = {row.key: 0 for row in records}

    for fold in range(6):
        splits = latin_square_semantic_split(records, fold=fold)
        assert {name: len(rows) for name, rows in splits.items()} == {
            "train": 108,
            "validation": 27,
            "test": 27,
        }
        keys = {name: {row.key for row in rows} for name, rows in splits.items()}
        assert not (keys["train"] & keys["validation"])
        assert not (keys["train"] & keys["test"])
        assert not (keys["validation"] & keys["test"])
        for key in keys["test"]:
            test_counts[key] += 1
    assert set(test_counts.values()) == {1}


def test_multilingual_paraphrases_inherit_pair_split_without_text_leakage(tmp_path):
    path = tmp_path / "paraphrases.json"
    _write_paraphrases(path)
    config = load_semantic_paraphrase_config(path)

    records, split_names = build_multilingual_semantic_records(_records(), config, fold=0)

    counts = {name: split_names.count(name) for name in ("train", "validation", "test")}
    assert counts == {"train": 540, "validation": 54, "test": 54}
    by_split = {
        name: [record for record, split in zip(records, split_names) if split == name]
        for name in counts
    }
    assert {record.source for record in by_split["train"]} == {
        "canonical_en",
        "paraphrase_zh_train",
    }
    assert {record.source for record in by_split["validation"]} == {
        "canonical_en",
        "paraphrase_zh_validation",
    }
    assert {record.source for record in by_split["test"]} == {
        "canonical_en",
        "paraphrase_zh_test",
    }
    normalized = {
        name: {" ".join(record.text.casefold().split()) for record in rows}
        for name, rows in by_split.items()
    }
    assert not (normalized["train"] & normalized["validation"])
    assert not (normalized["train"] & normalized["test"])
    assert not (normalized["validation"] & normalized["test"])


def test_deployment_records_train_all_label_pairs_and_hold_out_chinese_wording(tmp_path):
    path = tmp_path / "paraphrases.json"
    _write_paraphrases(path)
    config = load_semantic_paraphrase_config(path)

    records, split_names = build_deployment_semantic_records(_records(), config)

    counts = {name: split_names.count(name) for name in ("train", "validation", "test")}
    assert counts == {"train": 810, "validation": 162, "test": 162}
    train_keys = {record.key for record, split in zip(records, split_names) if split == "train"}
    assert len(train_keys) == 162
    for record, split in zip(records, split_names):
        if split != "train":
            assert record.language == "zh"
            assert record.source == f"paraphrase_zh_{split}"
    normalized = {
        name: {
            " ".join(record.text.casefold().split())
            for record, split in zip(records, split_names)
            if split == name
        }
        for name in ("train", "validation", "test")
    }
    assert not (normalized["train"] & normalized["validation"])
    assert not (normalized["train"] & normalized["test"])
    assert not (normalized["validation"] & normalized["test"])


def test_paraphrase_loader_rejects_missing_labels_and_reused_eval_phrase(tmp_path):
    path = tmp_path / "paraphrases.json"
    _write_paraphrases(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["behaviors"].pop(KIMODO_BEHAVIOR_IDS[-1])
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly all Kimodo behavior IDs"):
        load_semantic_paraphrase_config(path)

    _write_paraphrases(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    behavior = KIMODO_BEHAVIOR_IDS[0]
    payload["behaviors"][behavior]["test_phrase"] = payload["behaviors"][behavior]["train_phrases"][0]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="phrases overlap"):
        load_semantic_paraphrase_config(path)

    _write_paraphrases(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    first, second = KIMODO_BEHAVIOR_IDS[:2]
    payload["behaviors"][second]["test_phrase"] = payload["behaviors"][first]["validation_phrase"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="paraphrase text.*is reused"):
        load_semantic_paraphrase_config(path)

    _write_paraphrases(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    first, second = KIMODO_EMOTION_IDS[:2]
    payload["emotions"][second]["validation_prefix"] = payload["emotions"][first]["test_prefix"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="paraphrase text.*is reused"):
        load_semantic_paraphrase_config(path)


def test_last_token_pool_supports_left_and_right_padding():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    right_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    left_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

    right = last_token_pool(hidden, right_mask)
    left = last_token_pool(hidden, left_mask)

    assert torch.equal(right, torch.stack([hidden[0, 1], hidden[1, 2]]))
    assert torch.equal(left, hidden[:, -1])


def test_semantic_adapter_network_and_weighted_loss_are_finite():
    model = SemanticAdapterNetwork(embedding_dim=16, hidden_dim=8, dropout=0.0)
    output = model(torch.randn(4, 16))
    losses = semantic_adapter_loss(output, [0, 1, 2, 3], [0, 1, 2, 3])

    assert output["behavior_logits"].shape == (4, 27)
    assert output["emotion_logits"].shape == (4, 6)
    assert torch.isfinite(losses["total"])

    dual = SemanticAdapterNetwork(
        embedding_dim=16,
        hidden_dim=8,
        dropout=0.0,
        architecture="dual_task",
        component_embedding_dim=8,
    )
    dual_output = dual(torch.randn(4, 16))
    assert dual_output["behavior_logits"].shape == (4, 27)
    assert dual_output["emotion_logits"].shape == (4, 6)
    with pytest.raises(ValueError, match="two equal embedding components"):
        SemanticAdapterNetwork(
            embedding_dim=16,
            hidden_dim=8,
            architecture="dual_task",
            component_embedding_dim=7,
        )


def test_condition_builder_uses_prediction_and_allows_strict_explicit_overrides():
    calls = []

    def base_builder(text, **kwargs):
        calls.append((text, kwargs))
        return np.zeros(kwargs["condition_dim"], dtype=np.float32)

    builder = AdapterConditionBuilder(FixedSemanticAdapter(), base_builder=base_builder)
    condition = builder("开心地挥手", condition_dim=136)

    assert condition.shape == (136,)
    assert calls[-1][1]["behavior_id"] == "Behavior.GreetingOwner01"
    assert calls[-1][1]["emotion_id"] == "happy"
    assert builder.last_prediction.behavior_confidence == pytest.approx(0.8)

    builder(
        "开心地挥手",
        behavior_id="Behavior.GreetingOwner04",
        emotion_id="surprise",
        condition_dim=136,
    )
    assert calls[-1][1]["behavior_id"] == "Behavior.GreetingOwner04"
    assert builder.last_prediction.behavior_confidence == 1.0
    with pytest.raises(ValueError, match="136-dimensional"):
        builder("wave", condition_dim=92)
    with pytest.raises(ValueError, match="unknown Kimodo behavior_id"):
        builder("wave", behavior_id="invalid", condition_dim=136)


def test_condition_builder_uses_exact_training_condition_bank_instead_of_raw_text_hash():
    vectors = torch.zeros(len(KIMODO_BEHAVIOR_IDS), len(KIMODO_EMOTION_IDS), 136)
    behavior_index = KIMODO_BEHAVIOR_IDS.index("Behavior.GreetingOwner01")
    emotion_index = KIMODO_EMOTION_IDS.index("happy")
    vectors[behavior_index, emotion_index] = torch.arange(136, dtype=torch.float32)
    bank = {
        "contract_version": KIMODO_CONDITION_CONTRACT_VERSION,
        "condition_schema_version": KIMODO_CONDITION_SCHEMA_VERSION,
        "condition_dim": 136,
        "behavior_ids": list(KIMODO_BEHAVIOR_IDS),
        "emotion_ids": list(KIMODO_EMOTION_IDS),
        "vectors": vectors,
        "source_semantic_index": "fixture.parquet",
        "source_semantic_index_sha256": "0" * 64,
        "canonical_vectors_sha256": kimodo_condition_vectors_sha256(vectors.numpy()),
    }

    builder = AdapterConditionBuilder(
        FixedSemanticAdapter(),
        condition_bank=bank,
        base_builder=lambda *args, **kwargs: pytest.fail("canonical bank should bypass text builder"),
    )
    condition = builder("完全不同的中文自由文本", condition_dim=136)

    assert np.array_equal(condition, np.arange(136, dtype=np.float32))
    assert validate_condition_bank(bank) is bank
    tampered = dict(bank, vectors=bank["vectors"].clone())
    tampered["vectors"][behavior_index, emotion_index, 0] += 1
    with pytest.raises(ValueError, match="vector hash mismatch"):
        validate_condition_bank(tampered)


def test_semantic_checkpoint_roundtrip_with_injected_offline_encoder(tmp_path):
    model = SemanticAdapterNetwork(embedding_dim=33, hidden_dim=12, dropout=0.0)
    payload = semantic_adapter_checkpoint_payload(
        model,
        model_name="fake/qwen",
        model_revision="abc123",
        instruction="classify motion",
        max_length=64,
        fold=0,
        best_step=4,
        validation_metrics={"joint_accuracy": 0.5},
        test_metrics={"joint_accuracy": 0.4},
        split_keys={"train": [], "validation": [], "test": []},
        condition_bank=_condition_bank(),
    )
    path = tmp_path / "adapter.pt"
    torch.save(payload, path)

    adapter, checkpoint = load_semantic_adapter(
        path,
        text_encoder=FakeTextEncoder(),
        allow_incompatible_encoder=True,
        device="cpu",
    )
    prediction = adapter.predict_one("hello")

    assert prediction.behavior_id in KIMODO_BEHAVIOR_IDS
    assert prediction.emotion_id in KIMODO_EMOTION_IDS
    assert checkpoint["qwen"]["revision"] == "abc123"
    assert checkpoint["condition_bank"]["vectors"].shape == (27, 6, 136)
    assert checkpoint["condition_bank"]["source_semantic_index_sha256"] == "0" * 64
    bad = dict(payload, behavior_ids=list(reversed(KIMODO_BEHAVIOR_IDS)))
    with pytest.raises(ValueError, match="label order"):
        validate_semantic_adapter_checkpoint(bad)

    with pytest.raises(ValueError, match="does not match the encoder used to train"):
        load_semantic_adapter(
            path,
            model_name="different/qwen",
            text_encoder_factory=lambda **kwargs: FakeTextEncoder(),
            device="cpu",
        )

    adapter, _ = load_semantic_adapter(
        path,
        model_name="different/qwen",
        allow_incompatible_encoder=True,
        text_encoder_factory=lambda **kwargs: FakeTextEncoder(),
        device="cpu",
    )
    assert adapter.predict_one("override experiment").behavior_id in KIMODO_BEHAVIOR_IDS


def test_dual_task_checkpoint_roundtrip_preserves_two_instruction_encoder_contract(tmp_path):
    class FakeDualEncoder:
        embedding_dim = 34

        def encode(self, texts, *, batch_size=16):
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

    model = SemanticAdapterNetwork(
        embedding_dim=34,
        hidden_dim=12,
        dropout=0.0,
        architecture="dual_task",
        component_embedding_dim=17,
    )
    payload = semantic_adapter_checkpoint_payload(
        model,
        model_name="fake/qwen",
        model_revision="abc123",
        instruction="physical action",
        secondary_instruction="explicit emotion",
        component_embedding_dim=17,
        max_length=64,
        fold=0,
        best_step=4,
        validation_metrics={"joint_accuracy": 0.5},
        test_metrics={"joint_accuracy": 0.4},
        split_keys={"train": [], "validation": [], "test": []},
    )
    path = tmp_path / "dual_adapter.pt"
    torch.save(payload, path)

    adapter, checkpoint = load_semantic_adapter(
        path,
        text_encoder=FakeDualEncoder(),
        allow_incompatible_encoder=True,
        device="cpu",
    )

    assert adapter.network.architecture == "dual_task"
    assert checkpoint["qwen"]["instructions"] == ["physical action", "explicit emotion"]
    malformed = dict(payload, qwen=dict(payload["qwen"], instructions=["only one"]))
    with pytest.raises(ValueError, match="component metadata"):
        validate_semantic_adapter_checkpoint(malformed)


def test_training_head_uses_compositional_holdout_and_exports_small_checkpoint(tmp_path):
    records = _records()
    split_records = latin_square_semantic_split(records, fold=2)
    split_by_key = {row.key: name for name, rows in split_records.items() for row in rows}
    split_names = [split_by_key[row.key] for row in records]
    embeddings = np.zeros((len(records), 33), dtype=np.float32)
    for row, record in enumerate(records):
        embeddings[row, KIMODO_BEHAVIOR_IDS.index(record.behavior_id)] = 1.0
        embeddings[row, 27 + KIMODO_EMOTION_IDS.index(record.emotion_id)] = 1.0

    summary = train_semantic_adapter_head(
        records,
        embeddings,
        split_names,
        output_dir=tmp_path / "run",
        model_name="fake/qwen",
        model_revision="revision",
        hidden_dim=64,
        dropout=0.0,
        steps=300,
        batch_size=64,
        lr=0.02,
        eval_interval=10,
        early_stopping_patience=20,
        device="cpu",
    )

    run = tmp_path / "run"
    assert summary["validation"]["joint_accuracy"] >= 0.95
    assert summary["test"]["joint_accuracy"] >= 0.95
    assert summary["adapter_parameters"] < 100_000
    assert (run / "semantic_adapter_checkpoint.pt").is_file()
    assert json.loads((run / "metrics.json").read_text())["test"]["count"] == 27
    manifest = json.loads((run / "split_manifest.json").read_text())
    keys = {
        name: {(row["behavior_id"], row["emotion_id"]) for row in rows}
        for name, rows in manifest.items()
    }
    assert not (keys["train"] & keys["validation"])
    assert not (keys["train"] & keys["test"])


def test_evaluation_reports_macro_and_joint_accuracy():
    records = _records()[:6]
    embeddings = np.eye(6, 33, dtype=np.float32)
    model = SemanticAdapterNetwork(embedding_dim=33, hidden_dim=8, dropout=0.0)

    metrics = evaluate_semantic_adapter(model, embeddings, records)

    assert set(metrics) == {
        "loss",
        "behavior_accuracy",
        "emotion_accuracy",
        "joint_accuracy",
        "behavior_macro_accuracy",
        "emotion_macro_accuracy",
        "count",
    }
    assert metrics["count"] == 6


def test_training_config_is_strict_and_script_supports_direct_execution(tmp_path):
    args = training_args_from_config(
        {
            "prompt_csv": "prompts.csv",
            "output_dir": "run",
            "embedding_dim": 128,
            "paraphrases_json": "paraphrases.json",
            "condition_dataset_dir": "dataset",
            "local_files_only": True,
        }
    )
    assert args.embedding_dim == 128
    assert args.paraphrases_json == "paraphrases.json"
    assert args.condition_dataset_dir == "dataset"
    assert args.local_files_only is True
    with pytest.raises(ValueError, match="unknown semantic adapter config"):
        training_args_from_config({"prompt_csv": "x", "output_dir": "y", "unknown": 1})
    with pytest.raises(ValueError, match="configured together"):
        training_args_from_config(
            {
                "prompt_csv": "x",
                "output_dir": "y",
                "adapter_architecture": "dual_task",
            }
        )

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "training/scripts/train_semantic_adapter.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "frozen-Qwen" in completed.stdout
