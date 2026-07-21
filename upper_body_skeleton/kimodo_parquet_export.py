#!/usr/bin/env python3
import argparse
import csv
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from upper_body_skeleton.kimodo_semantics import KimodoPromptRecord, load_kimodo_prompt_index
from upper_body_skeleton.lerobot_export import export_lerobot_dataset
from upper_body_skeleton.retarget_v2 import JOINT_ORDER


DEFAULT_FPS = 30.0
DEFAULT_ROWS_PER_FILE = 250_000
_CSV_STEM_RE = re.compile(r"^(?P<behavior>.+?)__(?P<emotion>[a-z]+)__sample_(?P<sample>\d+)$")
AFFECT_DEFAULTS = {
    "neutral": (0.0, 0.0, 0, 0),
    "sad_like": (-0.25, -0.45, 0, 0),
    "nervous": (0.45, -0.2, 2, 0),
    "friendly": (0.25, 0.45, 1, 2),
    "uncertain": (0.2, -0.15, 1, 0),
    "angry_like": (0.65, -0.55, 3, 0),
    "excited": (0.75, 0.45, 3, 2),
}
INTENT_KEYWORDS = [
    ("warning", ("warning", "warn", "stop", "danger", "careful", "警告", "小心", "停止", "危险")),
    ("requesting_help", ("help", "assist", "please", "request", "求助", "帮忙", "请求", "请")),
    ("greeting", ("hello", "hi", "greet", "wave", "你好", "打招呼", "挥手")),
    ("refusing", ("refuse", "reject", "no ", "don't", "cannot", "拒绝", "不要", "不行")),
    ("explaining", ("explain", "tell", "describe", "conversational", "解释", "说明", "表达", "讲")),
]
AFFECT_KEYWORDS = [
    ("excited", ("excited", "energetic", "happy", "eager", "兴奋", "激动", "高兴")),
    ("angry_like", ("angry", "frustrated", "annoyed", "生气", "愤怒", "不满")),
    ("nervous", ("nervous", "tense", "anxious", "紧张", "焦虑")),
    ("uncertain", ("uncertain", "unsure", "hesitant", "不确定", "犹豫")),
    ("sad_like", ("sad", "down", "upset", "难过", "低落")),
    ("friendly", ("friendly", "warm", "kind", "友好", "亲切")),
]
STYLE_KEYWORDS = [
    ("energetic", ("energetic", "large", "fast", "big", "active", "用力", "大幅度", "快速")),
    ("relaxed", ("relaxed", "soft", "calm", "loose", "放松", "柔和", "平静")),
    ("restrained", ("restrained", "small", "subtle", "reserved", "克制", "小幅度", "保守")),
]
GESTURE_KEYWORDS = [
    ("crossed_arms", ("cross", "fold arms", "arms crossed", "交叉", "抱臂")),
    ("pointing", ("point", "pointing", "指", "指向")),
    ("shrugging", ("shrug", "shrugging", "耸肩")),
    ("waving", ("wave", "waving", "挥手", "摆手")),
    ("upper_body_gesture", ("gesture", "body", "upper body", "手势", "肢体", "上肢")),
]


@dataclass(frozen=True)
class ParsedKimodoCsvPath:
    path: Path
    behavior_slug: str
    emotion_id: str
    prompt_stem: str
    sample_index: int


def parse_kimodo_csv_path(csv_path, csv_root):
    csv_path = Path(csv_path)
    csv_root = Path(csv_root)
    stem = csv_path.stem
    match = _CSV_STEM_RE.match(stem)
    if not match:
        raise ValueError(f"Kimodo CSV filename must match '<behavior>__<emotion>__sample_<NN>': {csv_path}")
    behavior_slug = match.group("behavior")
    emotion_id = match.group("emotion")
    sample_index = int(match.group("sample"))

    try:
        relative = csv_path.relative_to(csv_root)
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) >= 3:
        dir_behavior, dir_emotion = relative.parts[-3], relative.parts[-2]
        if dir_behavior != behavior_slug or dir_emotion != emotion_id:
            raise ValueError(
                f"Kimodo CSV path disagrees with filename: path has {dir_behavior}/{dir_emotion}, "
                f"filename has {behavior_slug}/{emotion_id}"
            )

    return ParsedKimodoCsvPath(
        path=csv_path,
        behavior_slug=behavior_slug,
        emotion_id=emotion_id,
        prompt_stem=f"{behavior_slug}__{emotion_id}",
        sample_index=sample_index,
    )


def _load_prompt_stem_index(prompt_csv):
    records = load_kimodo_prompt_index(prompt_csv).values()
    index = {}
    for record in records:
        stem = Path(record.output_name).stem.lower()
        if stem in index:
            raise ValueError(f"Duplicate Kimodo prompt output stem: {stem}")
        index[stem] = record
    return index


def _count_csv_rows(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _duration_sec(frame_count, fps):
    return float(frame_count) / float(fps) if frame_count else 0.0


def _first_keyword_label(text, choices, default):
    lowered = f" {str(text).lower()} "
    for label, keywords in choices:
        if any(keyword in lowered for keyword in keywords):
            return label
    return default


def _infer_codes_from_text(text):
    return {
        "intent": _first_keyword_label(text, INTENT_KEYWORDS, "explaining"),
        "observed_affect": _first_keyword_label(text, AFFECT_KEYWORDS, "neutral"),
        "motion_style": _first_keyword_label(text, STYLE_KEYWORDS, "restrained"),
        "semantic_gesture": _first_keyword_label(text, GESTURE_KEYWORDS, "upper_body_gesture"),
    }


def _semantic_defaults(prompt_record: KimodoPromptRecord):
    codes = _infer_codes_from_text(prompt_record.prompt)
    affect = codes["observed_affect"]
    arousal, valence, arousal_token, valence_token = AFFECT_DEFAULTS.get(affect, AFFECT_DEFAULTS["neutral"])
    return {
        "intent": codes["intent"],
        "observed_affect": affect,
        "motion_style": codes["motion_style"],
        "semantic_gesture": codes["semantic_gesture"],
        "arousal": arousal,
        "valence": valence,
        "arousal_token": arousal_token,
        "valence_token": valence_token,
    }


def kimodo_record_for_csv(parsed, prompt_record, prompt_csv, fps=DEFAULT_FPS):
    frame_count = _count_csv_rows(parsed.path)
    duration = _duration_sec(frame_count, fps)
    defaults = _semantic_defaults(prompt_record)
    sample_id = parsed.path.stem
    prompt = prompt_record.prompt
    csv_path = str(parsed.path)
    return {
        "sample_id": sample_id,
        "source": {
            "dataset": "kimodo",
            "video_path": "",
            "json_path": "",
            "npz_path": "",
            "retarget_csv_path": csv_path,
            "prompt_csv_path": str(prompt_csv),
        },
        "time_window": {"start_sec": 0.0, "end_sec": duration, "fps": float(fps)},
        "language_condition": {
            "raw_transcript": "",
            "scenario_description": prompt,
            "action_description": prompt,
            "intent_text": defaults["intent"],
            "mood_text": prompt_record.emotion_zh_label or prompt_record.emotion_id,
            "rationale_text": prompt,
            "instruction_variants": [prompt],
        },
        "labels": {
            "intent": defaults["intent"],
            "communicative_intent": defaults["intent"],
            "observed_affect": defaults["observed_affect"],
            "gesture_function": "kimodo_behavior",
            "emotion_trajectory": f"{prompt_record.emotion_id}_sustained",
            "motion_style": defaults["motion_style"],
            "intensity": "medium",
            "speed": "normal",
            "openness": "neutral",
            "tension": "neutral",
            "duration_sec": duration,
            "transition": "end",
            "semantic_confidence": 1.0,
            "arousal": defaults["arousal"],
            "valence": defaults["valence"],
            "arousal_token": defaults["arousal_token"],
            "valence_token": defaults["valence_token"],
            "motion_energy": 0.05,
            "label_sources": ["kimodo_action_emotion_prompts.csv", "kimodo_csv_path"],
            "behavior_id": prompt_record.behavior_id,
            "emotion_id": prompt_record.emotion_id,
        },
        "meta_semantics": {
            "semantic_gesture": defaults["semantic_gesture"],
            "behavior_id": prompt_record.behavior_id,
            "emotion_id": prompt_record.emotion_id,
            "kimodo": {
                "behavior_slug": parsed.behavior_slug,
                "emotion_zh_label": prompt_record.emotion_zh_label,
                "negative_prompt": prompt_record.negative_prompt,
                "output_name": prompt_record.output_name,
                "output_format": prompt_record.output_format,
                "requires_bvh_without_t_pose": prompt_record.requires_bvh_without_t_pose,
                "sample_index": parsed.sample_index,
            },
            "body_expression": {
                "communicative_intent": defaults["intent"],
                "gesture_function": "kimodo_behavior",
                "emotion_trajectory": f"{prompt_record.emotion_id}_sustained",
                "intensity": "medium",
                "speed": "normal",
                "openness": "neutral",
                "tension": "neutral",
                "duration_sec": duration,
                "transition": "end",
            },
            "body_motion": {
                "joint_space": "v2_upper_body_15d",
                "action_shape": [frame_count, len(JOINT_ORDER)],
                "joint_order": JOINT_ORDER,
                "motion_style": defaults["motion_style"],
                "motion_energy": 0.05,
                "semantic_gesture": defaults["semantic_gesture"],
            },
        },
        "action": {
            "retarget_csv_path": csv_path,
            "start_row": 0,
            "end_row": frame_count,
            "fps": float(fps),
            "joint_order": JOINT_ORDER,
        },
        "quality": {
            "accepted_for_training": frame_count > 0,
            "frame_count": frame_count,
            "flagged_frame_count": 0,
            "max_elbow_overfold": 0.0,
            "max_yaw_under_response": 0.0,
        },
    }


def build_kimodo_jsonl(
    kimodo_root,
    output_jsonl,
    *,
    prompt_csv=None,
    csv_root=None,
    fps=DEFAULT_FPS,
    max_episodes=None,
    skip_missing_prompts=False,
):
    kimodo_root = Path(kimodo_root)
    prompt_csv = Path(prompt_csv) if prompt_csv else kimodo_root / "kimodo_action_emotion_prompts.csv"
    csv_root = Path(csv_root) if csv_root else kimodo_root / "csv"
    output_jsonl = Path(output_jsonl)
    prompt_index = _load_prompt_stem_index(prompt_csv)
    csv_paths = sorted(path for path in csv_root.glob("*/*/*.csv") if path.name != ".DS_Store")
    missing = []
    records = []

    for csv_path in csv_paths:
        parsed = parse_kimodo_csv_path(csv_path, csv_root)
        prompt_record = prompt_index.get(parsed.prompt_stem.lower())
        if prompt_record is None:
            missing.append(str(csv_path))
            if skip_missing_prompts:
                continue
            continue
        records.append(kimodo_record_for_csv(parsed, prompt_record, prompt_csv, fps=fps))
        if max_episodes is not None and len(records) >= int(max_episodes):
            break

    if missing and not skip_missing_prompts:
        preview = "\n".join(missing[:10])
        raise ValueError(f"Missing Kimodo prompt rows for {len(missing)} CSV files. First missing:\n{preview}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "records": len(records),
        "csv_files": len(csv_paths),
        "missing_prompts": len(missing),
        "jsonl_path": str(output_jsonl),
        "prompt_csv": str(prompt_csv),
        "csv_root": str(csv_root),
    }


def export_kimodo_parquet_dataset(
    kimodo_root,
    output_dir,
    *,
    prompt_csv=None,
    csv_root=None,
    fps=DEFAULT_FPS,
    rows_per_file=DEFAULT_ROWS_PER_FILE,
    max_episodes=None,
    skip_missing_prompts=False,
    keep_jsonl=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if keep_jsonl:
        jsonl_path = output_dir / "kimodo_language_action_index.jsonl"
        jsonl_summary = build_kimodo_jsonl(
            kimodo_root,
            jsonl_path,
            prompt_csv=prompt_csv,
            csv_root=csv_root,
            fps=fps,
            max_episodes=max_episodes,
            skip_missing_prompts=skip_missing_prompts,
        )
        parquet_summary = export_lerobot_dataset(jsonl_path, output_dir, rows_per_file=rows_per_file)
    else:
        with tempfile.TemporaryDirectory(prefix="kimodo_export_") as tmp:
            jsonl_path = Path(tmp) / "kimodo_language_action_index.jsonl"
            jsonl_summary = build_kimodo_jsonl(
                kimodo_root,
                jsonl_path,
                prompt_csv=prompt_csv,
                csv_root=csv_root,
                fps=fps,
                max_episodes=max_episodes,
                skip_missing_prompts=skip_missing_prompts,
            )
            parquet_summary = export_lerobot_dataset(jsonl_path, output_dir, rows_per_file=rows_per_file)

    summary = {"jsonl": jsonl_summary, "parquet": parquet_summary}
    summary_path = output_dir / "kimodo_export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export Kimodo CSV motions and prompts to ULA MMDiT-lite LeRobot parquet")
    parser.add_argument("--kimodo-root", default="Kimodo_CSV")
    parser.add_argument("--prompt-csv")
    parser.add_argument("--csv-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--rows-per-file", type=int, default=DEFAULT_ROWS_PER_FILE)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--skip-missing-prompts", action="store_true")
    parser.add_argument("--keep-jsonl", action="store_true")
    args = parser.parse_args()

    summary = export_kimodo_parquet_dataset(
        args.kimodo_root,
        args.output_dir,
        prompt_csv=args.prompt_csv,
        csv_root=args.csv_root,
        fps=args.fps,
        rows_per_file=args.rows_per_file,
        max_episodes=args.max_episodes,
        skip_missing_prompts=args.skip_missing_prompts,
        keep_jsonl=args.keep_jsonl,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
