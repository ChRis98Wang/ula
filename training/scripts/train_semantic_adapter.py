#!/usr/bin/env python3
import argparse
import json
from argparse import Namespace
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import yaml
except ImportError:  # pragma: no cover - JSON and direct CLI remain available
    yaml = None

from upper_body_skeleton.semantic_adapter import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_TASK_INSTRUCTION,
    FrozenQwenTextEncoder,
    build_deployment_semantic_records,
    build_multilingual_semantic_records,
    latin_square_semantic_split,
    load_semantic_paraphrase_config,
    load_semantic_prompt_catalog,
    load_kimodo_condition_bank,
    train_semantic_adapter_head,
)


DEFAULTS = {
    "model_name": DEFAULT_QWEN_MODEL,
    "revision": None,
    "instruction": DEFAULT_TASK_INSTRUCTION,
    "secondary_instruction": None,
    "embedding_dim": 256,
    "max_length": 256,
    "encode_batch_size": 16,
    "hidden_dim": 128,
    "dropout": 0.1,
    "steps": 1_000,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "eval_interval": 25,
    "early_stopping_patience": 20,
    "fold": 0,
    "seed": 7,
    "device": "auto",
    "local_files_only": False,
    "overwrite": False,
    "paraphrases_json": None,
    "adapter_architecture": "shared",
    "split_mode": "latin",
    "condition_dataset_dir": None,
}


def load_train_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML configs; use JSON or direct CLI options")
    return yaml.safe_load(text) or {}


def _required(config, key):
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"semantic adapter training config requires {key}")
    return value


def training_args_from_config(config):
    allowed = set(DEFAULTS) | {"prompt_csv", "output_dir"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown semantic adapter config keys: {unknown}")
    values = DEFAULTS | {key: config[key] for key in DEFAULTS if key in config}
    if values["adapter_architecture"] not in {"shared", "dual_task"}:
        raise ValueError(f"unknown semantic adapter architecture: {values['adapter_architecture']}")
    if values["split_mode"] not in {"latin", "deployment"}:
        raise ValueError(f"unknown semantic adapter split mode: {values['split_mode']}")
    has_secondary_instruction = values["secondary_instruction"] not in (None, "")
    if (values["adapter_architecture"] == "dual_task") != has_secondary_instruction:
        raise ValueError("dual_task adapter architecture and secondary_instruction must be configured together")
    return Namespace(
        prompt_csv=str(_required(config, "prompt_csv")),
        paraphrases_json=(
            None if values["paraphrases_json"] in (None, "") else str(values["paraphrases_json"])
        ),
        output_dir=str(_required(config, "output_dir")),
        condition_dataset_dir=(
            None if values["condition_dataset_dir"] in (None, "") else str(values["condition_dataset_dir"])
        ),
        model_name=str(values["model_name"]),
        revision=None if values["revision"] in (None, "") else str(values["revision"]),
        instruction=str(values["instruction"]),
        secondary_instruction=(
            None if values["secondary_instruction"] in (None, "") else str(values["secondary_instruction"])
        ),
        embedding_dim=int(values["embedding_dim"]),
        max_length=int(values["max_length"]),
        encode_batch_size=int(values["encode_batch_size"]),
        hidden_dim=int(values["hidden_dim"]),
        dropout=float(values["dropout"]),
        steps=int(values["steps"]),
        batch_size=int(values["batch_size"]),
        lr=float(values["lr"]),
        weight_decay=float(values["weight_decay"]),
        eval_interval=int(values["eval_interval"]),
        early_stopping_patience=int(values["early_stopping_patience"]),
        fold=int(values["fold"]),
        seed=int(values["seed"]),
        device=str(values["device"]),
        local_files_only=bool(values["local_files_only"]),
        overwrite=bool(values["overwrite"]),
        adapter_architecture=str(values["adapter_architecture"]),
        split_mode=str(values["split_mode"]),
    )


def parse_training_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a frozen-Qwen Kimodo semantic classification adapter")
    parser.add_argument("--config")
    parser.add_argument("--prompt-csv")
    parser.add_argument("--paraphrases-json")
    parser.add_argument("--output-dir")
    parser.add_argument("--condition-dataset-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--instruction")
    parser.add_argument("--secondary-instruction")
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--adapter-architecture", choices=("shared", "dual_task"))
    parser.add_argument("--split-mode", choices=("latin", "deployment"))
    cli = parser.parse_args(argv)

    config = load_train_config(cli.config) if cli.config else {}
    for key, value in vars(cli).items():
        if key != "config" and value is not None:
            config[key] = value
    return training_args_from_config(config)


def main(argv=None):
    args = parse_training_args(argv)
    canonical_records = load_semantic_prompt_catalog(args.prompt_csv)
    if args.paraphrases_json:
        paraphrases = load_semantic_paraphrase_config(args.paraphrases_json)
        if args.split_mode == "deployment":
            records, split_names = build_deployment_semantic_records(canonical_records, paraphrases)
        else:
            records, split_names = build_multilingual_semantic_records(
                canonical_records,
                paraphrases,
                fold=args.fold,
            )
    else:
        if args.split_mode != "latin":
            raise ValueError("deployment split mode requires paraphrases_json")
        records = canonical_records
        split_records = latin_square_semantic_split(records, fold=args.fold)
        split_by_key = {
            record.key: split_name
            for split_name, rows in split_records.items()
            for record in rows
        }
        split_names = [split_by_key[record.key] for record in records]

    encoder = FrozenQwenTextEncoder(
        args.model_name,
        revision=args.revision,
        instruction=args.instruction,
        secondary_instruction=args.secondary_instruction,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    embeddings = encoder.encode([record.text for record in records], batch_size=args.encode_batch_size)
    condition_bank = (
        None if args.condition_dataset_dir is None else load_kimodo_condition_bank(args.condition_dataset_dir)
    )
    summary = train_semantic_adapter_head(
        records,
        embeddings,
        split_names,
        output_dir=args.output_dir,
        model_name=encoder.model_name,
        model_revision=encoder.revision,
        instruction=encoder.instruction,
        max_length=encoder.max_length,
        fold=args.fold,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        architecture=args.adapter_architecture,
        component_embedding_dim=encoder.component_embedding_dim,
        secondary_instruction=encoder.secondary_instruction,
        split_mode=args.split_mode,
        condition_bank=condition_bank,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
