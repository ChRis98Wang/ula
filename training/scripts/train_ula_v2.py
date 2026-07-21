#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from upper_body_skeleton.ula_training_v2 import resolve_v2_config, train_ula_v2


def load_config(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ULA V2 YAML config must contain a mapping")
    return resolve_v2_config(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the leakage-safe, variable-duration Kimodo ULA MMDiT V2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
        return
    train_ula_v2(config)


if __name__ == "__main__":
    main()
