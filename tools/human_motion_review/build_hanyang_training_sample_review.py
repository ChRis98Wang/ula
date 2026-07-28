#!/usr/bin/env python3
"""Build a human-review reel before Hanyang data may enter V8 training.

The reel shows the official 19-point source motion beside the source-faithful
ULA partial-18D retarget.  Every intended emotion contributes two train-split
examples: the best available human agreement and a distinct high-motion sample.
Rendering never grants training admission; the approval state remains false
until the user explicitly accepts the bundle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upper_body_skeleton.data_source_registry import (  # noqa: E402
    assert_no_forbidden_data_lineage,
)
from upper_body_skeleton.hanyang_emotion_retarget import (  # noqa: E402
    DATASET_ID,
    EMOTION_BY_ID,
    SOURCE_FPS,
    SOURCE_FRAMES,
    SOURCE_JOINTS,
    json_hash,
    load_hanyang_csv,
    sha256_file,
)
from upper_body_skeleton.retarget_v2_18d import JOINT_ORDER_18D  # noqa: E402


DEFAULT_RESEARCH_ROOT = Path(
    "/home/gez/nas/cloud/gez/human_motion/external_emotion_research/"
    "hanyang_emotional_body_motion_zenodo_10052504_v1"
)
DEFAULT_PASSED_MANIFEST = (
    DEFAULT_RESEARCH_ROOT / "retarget_v1" / "passed_manifest.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_RESEARCH_ROOT / "human_review" / "training_sample_v1"
)
DEFAULT_CROSS_SPLIT_REVIEW_MANIFEST = (
    DEFAULT_RESEARCH_ROOT
    / "retarget_v1"
    / "experimental_pool_v8"
    / "human_review_bundle"
    / "manifest.jsonl"
)
DEFAULT_URDF = (
    PROJECT_ROOT
    / "urdf_V2_20260514/urdf/xacro/robot_modify_meshdir.urdf"
)
DEFAULT_RENDERER_PYTHON = Path(
    "/home/gez/miniconda3/envs/env_isaaclab/bin/python"
)
EXPECTED_PASSED_MANIFEST_SHA256 = (
    "be9dcf53f0aa2acc2695475e8625d1f05550a07fb8403e46d0e0c8e3f633daab"
)
EXPECTED_CROSS_SPLIT_REVIEW_SHA256 = (
    "aafee677a3394102dbea5d35fb3f6c8b6c86ac5dd4f6ac7faccac75b9bce7c3a"
)
ARTIFACT_KIND = "hanyang_training_sample_human_review_bundle_v1"
QUEUE_ARTIFACT_KIND = "hanyang_training_sample_review_item_v1"
APPROVAL_ARTIFACT_KIND = "hanyang_training_sample_approval_gate_v1"
SAMPLE_POLICY = (
    "train_split_two_per_intended_emotion_best_human_agreement_"
    "and_distinct_highest_qc_pass_jerk_v1"
)
EMOTION_ORDER = tuple(EMOTION_BY_ID[index] for index in sorted(EMOTION_BY_ID))
SOURCE_WIDTH = 800
ROBOT_WIDTH = 640
PANEL_WIDTH = 480
HEIGHT = 720
OUTPUT_WIDTH = SOURCE_WIDTH + ROBOT_WIDTH + PANEL_WIDTH
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

SOURCE_EDGES = (
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Neck"),
    ("Neck", "Head"),
    ("Spine1", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine1", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
)
UPPER_BODY_JOINTS = frozenset(SOURCE_JOINTS[:13])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(stable_json(dict(row)) + "\n" for row in rows),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passed-manifest", type=Path, default=DEFAULT_PASSED_MANIFEST
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        default=EXPECTED_PASSED_MANIFEST_SHA256,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cross-split-review-manifest",
        type=Path,
        default=DEFAULT_CROSS_SPLIT_REVIEW_MANIFEST,
    )
    parser.add_argument(
        "--expected-cross-split-review-sha256",
        default=EXPECTED_CROSS_SPLIT_REVIEW_SHA256,
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--renderer-python", type=Path, default=DEFAULT_RENDERER_PYTHON
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(FONT_PATH)
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _load_quality(row: Mapping[str, Any], *, retarget_root: Path) -> dict[str, Any]:
    expected_row_hash = json_hash(
        {
            key: value
            for key, value in row.items()
            if key != "record_sha256"
        }
    )
    if (
        row.get("record_sha256") != expected_row_hash
        or row.get("dataset_id") != DATASET_ID
        or row.get("status") != "passed"
        or row.get("fixed_split_assignment") != "train"
        or row.get("kimodo_accessed_or_used") is not False
        or row.get("generator_foundation_eligible") is not False
    ):
        raise ValueError("review input is not an isolated train-split QC pass")
    quality_path = Path(str(row.get("quality_json"))).resolve()
    try:
        quality_path.relative_to(retarget_root)
    except ValueError as error:
        raise ValueError("quality path escapes retarget root") from error
    if (
        not quality_path.is_file()
        or sha256_file(quality_path) != row.get("quality_json_sha256")
    ):
        raise ValueError("quality report hash mismatch")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    expected_record = json_hash(
        {
            key: value
            for key, value in quality.items()
            if key != "record_sha256"
        }
    )
    if (
        quality.get("record_sha256") != expected_record
        or quality.get("record_sha256") != row.get("quality_record_sha256")
        or quality.get("clip_id") != row.get("clip_id")
        or quality.get("fixed_split_assignment") != "train"
        or (quality.get("quality_gate") or {}).get("passed") is not True
    ):
        raise ValueError("quality report identity or gate mismatch")
    return quality


def _candidate_record(
    row: Mapping[str, Any], quality: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = quality.get("emotion_evaluation") or {}
    trajectory = quality.get("trajectory") or {}
    majority = evaluation.get("unique_majority_emotion_id")
    if not majority:
        majority_values = evaluation.get("majority_emotion_ids") or []
        majority = "/".join(str(value) for value in majority_values) or "unknown"
    high_confidence_label_audit_only = bool(
        evaluation.get("rater_coverage_pass") is True
        and evaluation.get("intended_majority_agrees") is True
        and float(evaluation.get("intended_share", 0.0)) >= 0.70
    )
    return {
        "row": dict(row),
        "quality": dict(quality),
        "clip_id": str(row["clip_id"]),
        "source_stem": str(row["source_stem"]),
        "participant_id": int(row["participant_id"]),
        "intended_emotion": str(row["emotion_id"]),
        "human_majority": majority,
        "intended_share": float(evaluation.get("intended_share", 0.0)),
        "majority_share": float(evaluation.get("majority_share", 0.0)),
        "rater_count": int(evaluation.get("rater_count", 0)),
        "intended_majority_agrees": bool(
            evaluation.get("intended_majority_agrees", False)
        ),
        "intended_high_confidence": bool(
            evaluation.get("intended_high_confidence", False)
        ),
        "high_confidence_label_audit_only": (
            high_confidence_label_audit_only
        ),
        # The current V8 experiment is motion-only for every Hanyang row,
        # including rows whose source label happens to have high agreement.
        "emotion_condition_eligible": False,
        "rms_jerk_rad_s3": float(trajectory.get("rms_jerk_rad_s3", 0.0)),
        "max_velocity_limit_ratio": float(
            trajectory.get("max_velocity_limit_ratio", 0.0)
        ),
        "target_error_p95_m": float(
            quality.get("limb_target_error_p95_m", 0.0)
        ),
        "collision_frame_rate": float(
            quality.get("upper_body_collision_frame_rate", 0.0)
        ),
    }


def select_review_samples(
    rows: list[dict[str, Any]], *, retarget_root: Path
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("fixed_split_assignment") != "train":
            continue
        quality = _load_quality(row, retarget_root=retarget_root)
        candidate = _candidate_record(row, quality)
        grouped[candidate["intended_emotion"]].append(candidate)
    if set(grouped) != set(EMOTION_ORDER):
        raise ValueError("train QC-pass pool does not cover all intended emotions")

    selected: list[dict[str, Any]] = []
    for emotion in EMOTION_ORDER:
        candidates = grouped[emotion]
        best = max(
            candidates,
            key=lambda item: (
                item["high_confidence_label_audit_only"],
                item["intended_high_confidence"],
                item["intended_majority_agrees"],
                item["intended_share"],
                item["majority_share"],
                item["clip_id"],
            ),
        )
        remaining = [
            item
            for item in candidates
            if item["participant_id"] != best["participant_id"]
        ]
        if not remaining:
            remaining = [item for item in candidates if item is not best]
        if not remaining:
            raise ValueError(f"not enough distinct {emotion} review candidates")
        dynamic = max(
            remaining,
            key=lambda item: (
                item["rms_jerk_rad_s3"],
                item["max_velocity_limit_ratio"],
                item["clip_id"],
            ),
        )
        for role, item in (
            ("best_human_agreement", best),
            ("highest_motion_energy_distinct_participant", dynamic),
        ):
            projected = dict(item)
            projected["selection_role"] = role
            selected.append(projected)
    if len(selected) != 2 * len(EMOTION_ORDER):
        raise ValueError("review selection must contain exactly two per emotion")
    return selected


def _projection(
    values: np.ndarray,
    *,
    horizontal_index: int,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal = values[..., horizontal_index]
    vertical = values[..., 1]
    horizontal_center = 0.5 * (float(horizontal.min()) + float(horizontal.max()))
    vertical_center = 0.5 * (float(vertical.min()) + float(vertical.max()))
    horizontal_span = max(0.1, float(horizontal.max() - horizontal.min()))
    vertical_span = max(0.1, float(vertical.max() - vertical.min()))
    scale = 0.88 * min((x1 - x0) / horizontal_span, (y1 - y0) / vertical_span)
    screen_x = 0.5 * (x0 + x1) + (horizontal - horizontal_center) * scale
    screen_y = 0.5 * (y0 + y1) - (vertical - vertical_center) * scale
    return screen_x, screen_y


def source_skeleton_frames(positions: np.ndarray) -> Iterable[Image.Image]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (SOURCE_FRAMES, len(SOURCE_JOINTS), 3):
        raise ValueError("source positions must have shape [150,19,3]")
    index = {name: offset for offset, name in enumerate(SOURCE_JOINTS)}
    front_x, front_y = _projection(
        positions,
        horizontal_index=0,
        x0=30,
        x1=385,
        y0=68,
        y1=690,
    )
    side_x, side_y = _projection(
        positions,
        horizontal_index=2,
        x0=415,
        x1=770,
        y0=68,
        y1=690,
    )
    title_font = _font(25)
    small_font = _font(18)
    for frame_index in range(SOURCE_FRAMES):
        image = Image.new("RGB", (SOURCE_WIDTH, HEIGHT), (13, 18, 27))
        draw = ImageDraw.Draw(image)
        draw.text(
            (20, 15),
            "原始数据 / ORIGINAL 19-POINT MOTION",
            fill=(240, 244, 252),
            font=title_font,
        )
        draw.text((158, 48), "FRONT", fill=(148, 172, 205), font=small_font)
        draw.text((552, 48), "SIDE", fill=(148, 172, 205), font=small_font)
        draw.line((400, 58, 400, 700), fill=(53, 66, 84), width=2)
        for first, second in SOURCE_EDGES:
            upper = first in UPPER_BODY_JOINTS and second in UPPER_BODY_JOINTS
            color = (74, 214, 231) if upper else (110, 123, 145)
            width = 5 if upper else 3
            a = index[first]
            b = index[second]
            draw.line(
                (
                    float(front_x[frame_index, a]),
                    float(front_y[frame_index, a]),
                    float(front_x[frame_index, b]),
                    float(front_y[frame_index, b]),
                ),
                fill=color,
                width=width,
            )
            draw.line(
                (
                    float(side_x[frame_index, a]),
                    float(side_y[frame_index, a]),
                    float(side_x[frame_index, b]),
                    float(side_y[frame_index, b]),
                ),
                fill=color,
                width=width,
            )
        for joint_index, joint_name in enumerate(SOURCE_JOINTS):
            upper = joint_name in UPPER_BODY_JOINTS
            color = (255, 203, 92) if upper else (151, 161, 180)
            radius = 5 if upper else 4
            for screen_x, screen_y in (
                (
                    front_x[frame_index, joint_index],
                    front_y[frame_index, joint_index],
                ),
                (
                    side_x[frame_index, joint_index],
                    side_y[frame_index, joint_index],
                ),
            ):
                draw.ellipse(
                    (
                        float(screen_x - radius),
                        float(screen_y - radius),
                        float(screen_x + radius),
                        float(screen_y + radius),
                    ),
                    fill=color,
                )
        draw.text(
            (20, 692),
            f"frame {frame_index + 1:03d}/150   "
            f"t={frame_index / SOURCE_FPS:4.2f}s   30 Hz",
            fill=(185, 197, 216),
            font=small_font,
            anchor="ls",
        )
        yield image


def encode_frames(
    frames: Iterable[Image.Image],
    *,
    output: Path,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(SOURCE_FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-frames:v",
        str(frame_count),
        "-y",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    written = 0
    try:
        for image in frames:
            if image.size != (width, height) or image.mode != "RGB":
                raise ValueError("encoded frame shape/mode changed")
            process.stdin.write(image.tobytes())
            written += 1
    finally:
        process.stdin.close()
    returncode = process.wait()
    if returncode != 0 or written != frame_count:
        raise RuntimeError(
            f"source frame encoder failed: return={returncode}, frames={written}"
        )
    os.replace(temporary, output)


def panel_image(sample: Mapping[str, Any], *, sample_index: int) -> Image.Image:
    image = Image.new("RGB", (PANEL_WIDTH, HEIGHT), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    title_font = _font(27)
    header_font = _font(22)
    body_font = _font(19)
    small_font = _font(16)
    accent = (13, 94, 128)
    warning = (177, 63, 48)
    draw.rectangle((0, 0, PANEL_WIDTH, 72), fill=(24, 39, 57))
    draw.text(
        (20, 16),
        f"训练前人工检查  {sample_index:02d}/14",
        fill=(248, 250, 253),
        font=title_font,
    )
    y = 92

    def line(label: str, value: str, *, color=(32, 42, 54), gap=32) -> None:
        nonlocal y
        draw.text((22, y), label, fill=(91, 104, 120), font=small_font)
        draw.text((188, y - 3), value, fill=color, font=body_font)
        y += gap

    draw.text(
        (20, y),
        str(sample["clip_id"]),
        fill=accent,
        font=header_font,
    )
    y += 44
    line("源端 intended", str(sample["intended_emotion"]).upper())
    majority_color = (
        (34, 133, 88)
        if sample["intended_majority_agrees"]
        else warning
    )
    line(
        "人评 majority",
        str(sample["human_majority"]).upper(),
        color=majority_color,
    )
    line(
        "intended 得票",
        f"{100.0 * sample['intended_share']:.1f}%"
        f" / {sample['rater_count']} raters",
    )
    line(
        "训练通道",
        "PARTIAL MOTION-ONLY",
        color=warning,
    )
    line(
        "标签审计",
        (
            "HIGH-CONFIDENCE (AUDIT ONLY)"
            if sample["high_confidence_label_audit_only"]
            else "LOW/CONFLICT (AUDIT ONLY)"
        ),
        color=(
            (34, 133, 88)
            if sample["high_confidence_label_audit_only"]
            else warning
        ),
    )
    role_label = {
        "best_human_agreement": "最高人评一致度",
        "highest_motion_energy_distinct_participant": (
            "高动作强度（不同参与者）"
        ),
    }[str(sample["selection_role"])]
    line("抽样角色", role_label, gap=36)
    draw.line((20, y, PANEL_WIDTH - 20, y), fill=(195, 203, 214), width=2)
    y += 18
    draw.text((20, y), "机器人 QC（全部已通过）", fill=accent, font=header_font)
    y += 42
    line("jerk RMS", f"{sample['rms_jerk_rad_s3']:.1f} rad/s³")
    line("最大限速比", f"{sample['max_velocity_limit_ratio']:.3f}")
    line("目标误差 p95", f"{1000 * sample['target_error_p95_m']:.2f} mm")
    line("碰撞帧率", f"{100 * sample['collision_frame_rate']:.1f}%")
    line("split / participant", f"train / P{sample['participant_id']:02d}")
    y += 4
    draw.rounded_rectangle(
        (18, y, PANEL_WIDTH - 18, y + 94),
        radius=10,
        fill=(255, 240, 225),
        outline=(222, 162, 104),
        width=2,
    )
    draw.text(
        (32, y + 12),
        "未观测并永久 loss-mask：",
        fill=(126, 77, 34),
        font=small_font,
    )
    draw.text(
        (32, y + 40),
        "双腕 roll/pitch（4 DOF）+ head yaw",
        fill=(92, 57, 30),
        font=body_font,
    )
    draw.text(
        (32, y + 70),
        "IK: window-5 smooth, 保端点；无重定时",
        fill=(126, 77, 34),
        font=small_font,
    )
    draw.rectangle((0, HEIGHT - 54, PANEL_WIDTH, HEIGHT), fill=(145, 42, 35))
    draw.text(
        (PANEL_WIDTH // 2, HEIGHT - 28),
        "尚未批准训练 / NOT APPROVED",
        fill=(255, 255, 255),
        font=header_font,
        anchor="mm",
    )
    return image


def render_robot(
    trajectory: Path,
    *,
    output: Path,
    summary: Path,
    renderer_python: Path,
    urdf: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_python),
        "-m",
        "upper_body_skeleton.mujoco_playback",
        "--joint-csv",
        str(trajectory),
        "--output-mp4",
        str(output),
        "--summary-json",
        str(summary),
        "--fps",
        str(SOURCE_FPS),
        "--width",
        str(ROBOT_WIDTH),
        "--height",
        str(HEIGHT),
        "--camera-margin",
        "1.12",
        "--camera-lookat-z-offset",
        "-0.06",
        "--urdf",
        str(urdf),
    ]
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"
    environment["PYOPENGL_PLATFORM"] = "egl"
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MuJoCo renderer failed:\n"
            + completed.stdout[-3000:]
            + completed.stderr[-3000:]
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if (
        payload.get("frames") != SOURCE_FRAMES
        or payload.get("joint_order") != list(JOINT_ORDER_18D)
        or payload.get("action_dim") != 18
    ):
        raise ValueError("MuJoCo render summary contract mismatch")


def compose_segment(
    source_video: Path,
    robot_video: Path,
    panel_png: Path,
    *,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-i",
        str(robot_video),
        "-loop",
        "1",
        "-framerate",
        str(SOURCE_FPS),
        "-i",
        str(panel_png),
        "-filter_complex",
        (
            f"[0:v]scale={SOURCE_WIDTH}:{HEIGHT},setsar=1[src];"
            f"[1:v]scale={ROBOT_WIDTH}:{HEIGHT},setsar=1[robot];"
            f"[2:v]scale={PANEL_WIDTH}:{HEIGHT},setsar=1[panel];"
            "[src][robot][panel]hstack=inputs=3[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-frames:v",
        str(SOURCE_FRAMES),
        "-y",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"review segment compose failed: {completed.stderr[-3000:]}")


def concat_segments(segments: list[Path], *, output: Path, root: Path) -> None:
    concat_path = root / "segments.concat.txt"
    atomic_text(
        concat_path,
        "".join(f"file '{path.resolve()}'\n" for path in segments),
    )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-an",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"review reel concat failed: {completed.stderr[-3000:]}")
    os.replace(temporary, output)


def _stream_duration(path: Path) -> float:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-i",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    marker = "Duration: "
    line = next(
        (value for value in completed.stderr.splitlines() if marker in value),
        "",
    )
    if not line:
        raise ValueError("ffmpeg did not report review reel duration")
    stamp = line.split(marker, 1)[1].split(",", 1)[0].strip()
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    passed_manifest = args.passed_manifest.resolve()
    output_root = args.output_root.resolve()
    urdf = args.urdf.resolve()
    renderer_python = args.renderer_python.resolve()
    cross_split_review = args.cross_split_review_manifest.resolve()
    assert_no_forbidden_data_lineage(
        {
            "manifest_path": str(passed_manifest),
            "output_path": str(output_root),
            "urdf_path": str(urdf),
            "source_manifest": str(cross_split_review),
        },
        context="hanyang_training_sample_review",
    )
    if sha256_file(passed_manifest) != args.expected_manifest_sha256:
        raise ValueError("completed Hanyang passed manifest hash changed")
    if (
        not cross_split_review.is_file()
        or sha256_file(cross_split_review)
        != args.expected_cross_split_review_sha256
    ):
        raise ValueError("cross-split review manifest hash changed")
    if not urdf.is_file() or not renderer_python.is_file():
        raise FileNotFoundError("renderer Python or ULA URDF is missing")
    retarget_root = passed_manifest.parent.resolve()
    rows = read_jsonl(passed_manifest)
    selected = select_review_samples(rows, retarget_root=retarget_root)
    queue: list[dict[str, Any]] = []
    segments: list[Path] = []
    for sample_index, sample in enumerate(selected, 1):
        quality = sample["quality"]
        stem = sample["source_stem"]
        item_root = output_root / "items" / f"{sample_index:02d}_{stem}"
        source_video = item_root / "source_19point.mp4"
        robot_video = item_root / "robot_source_faithful_partial18d.mp4"
        robot_summary = item_root / "robot_render_summary.json"
        panel_png = item_root / "review_panel.png"
        segment = output_root / "segments" / f"{sample_index:02d}_{stem}.mp4"

        source = load_hanyang_csv(quality["source_csv"])
        if source["source_sha256"] != quality["source_sha256"]:
            raise ValueError(f"{stem}: source CSV hash mismatch")
        faithful = Path(
            quality["outputs"]["source_faithful_partial_18d_csv"]
        ).resolve()
        if (
            sha256_file(faithful)
            != quality["outputs"][
                "source_faithful_partial_18d_csv_sha256"
            ]
        ):
            raise ValueError(f"{stem}: source-faithful CSV hash mismatch")
        encode_frames(
            source_skeleton_frames(source["positions"]),
            output=source_video,
            width=SOURCE_WIDTH,
            height=HEIGHT,
            frame_count=SOURCE_FRAMES,
        )
        render_robot(
            faithful,
            output=robot_video,
            summary=robot_summary,
            renderer_python=renderer_python,
            urdf=urdf,
        )
        panel = panel_image(sample, sample_index=sample_index)
        panel_png.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_png)
        compose_segment(
            source_video, robot_video, panel_png, output=segment
        )
        segments.append(segment)
        queue_row = {
            "schema_version": "1.0.0",
            "artifact_kind": QUEUE_ARTIFACT_KIND,
            "review_index": sample_index,
            "clip_id": sample["clip_id"],
            "source_stem": stem,
            "participant_id": sample["participant_id"],
            "fixed_split_assignment": "train",
            "intended_emotion": sample["intended_emotion"],
            "human_majority_emotion": sample["human_majority"],
            "intended_share": sample["intended_share"],
            "majority_share": sample["majority_share"],
            "intended_majority_agrees": sample[
                "intended_majority_agrees"
            ],
            "intended_high_confidence": sample[
                "intended_high_confidence"
            ],
            "high_confidence_label_audit_only": sample[
                "high_confidence_label_audit_only"
            ],
            "emotion_condition_eligible": False,
            "selection_role": sample["selection_role"],
            "training_lane": "partial_motion_only",
            "rms_jerk_rad_s3": sample["rms_jerk_rad_s3"],
            "max_velocity_limit_ratio": sample[
                "max_velocity_limit_ratio"
            ],
            "target_error_p95_m": sample["target_error_p95_m"],
            "collision_frame_rate": sample["collision_frame_rate"],
            "source_csv": quality["source_csv"],
            "source_csv_sha256": quality["source_sha256"],
            "source_faithful_partial_18d_csv": str(faithful),
            "source_faithful_partial_18d_csv_sha256": quality["outputs"][
                "source_faithful_partial_18d_csv_sha256"
            ],
            "quality_json": sample["row"]["quality_json"],
            "quality_json_sha256": sample["row"]["quality_json_sha256"],
            "source_video": str(source_video),
            "source_video_sha256": sha256_file(source_video),
            "robot_video": str(robot_video),
            "robot_video_sha256": sha256_file(robot_video),
            "review_segment": str(segment),
            "review_segment_sha256": sha256_file(segment),
            "manual_review_required": True,
            "accepted_for_training": False,
            "render_pass_grants_training_admission": False,
        }
        queue_row["record_sha256"] = json_hash(queue_row)
        queue.append(queue_row)

    queue_path = output_root / "review_queue.jsonl"
    atomic_jsonl(queue_path, queue)
    reel = output_root / "hanyang_training_sample_review_70s.mp4"
    concat_segments(segments, output=reel, root=output_root)
    duration = _stream_duration(reel)
    if not 69.5 <= duration <= 70.5:
        raise ValueError(f"review reel duration changed: {duration}")
    approval = {
        "schema_version": "1.0.0",
        "artifact_kind": APPROVAL_ARTIFACT_KIND,
        "review_bundle": str(output_root / "bundle_receipt.json"),
        "review_queue": str(queue_path),
        "review_queue_sha256": sha256_file(queue_path),
        "review_reel": str(reel),
        "review_reel_sha256": sha256_file(reel),
        "cross_split_review_manifest": str(cross_split_review),
        "cross_split_review_manifest_sha256": (
            args.expected_cross_split_review_sha256
        ),
        "sample_count": len(queue),
        "proposed_clip_ids": [row["clip_id"] for row in queue],
        "accepted_clip_ids": [],
        "rejected_clip_ids": [],
        "required_decision": (
            "explicit_user_approve_or_reject_after_visual_review"
        ),
        "human_review_required": True,
        "human_review_approved": False,
        "training_launch_allowed": False,
        "reviewed_by": None,
        "reviewed_utc": None,
        "decision_notes": None,
    }
    approval["record_sha256"] = json_hash(approval)
    approval_path = output_root / "approval_gate.json"
    atomic_json(approval_path, approval)
    receipt = {
        "schema_version": "1.0.0",
        "artifact_kind": ARTIFACT_KIND,
        "created_utc": utc_now(),
        "dataset_id": DATASET_ID,
        "passed_manifest": str(passed_manifest),
        "passed_manifest_sha256": args.expected_manifest_sha256,
        "sample_policy": SAMPLE_POLICY,
        "sample_count": len(queue),
        "samples_per_intended_emotion": 2,
        "source_view": "official_19_point_front_and_side_30hz",
        "robot_view": "source_faithful_partial_18d_no_time_dilation_30hz",
        "reel": str(reel),
        "reel_sha256": sha256_file(reel),
        "reel_duration_sec": duration,
        "reel_frames": len(queue) * SOURCE_FRAMES,
        "width": OUTPUT_WIDTH,
        "height": HEIGHT,
        "review_queue": str(queue_path),
        "review_queue_sha256": sha256_file(queue_path),
        "cross_split_review_manifest": str(cross_split_review),
        "cross_split_review_manifest_sha256": (
            args.expected_cross_split_review_sha256
        ),
        "approval_gate": str(approval_path),
        "approval_gate_sha256": sha256_file(approval_path),
        "emotion_condition_eligible_sample_count": 0,
        "high_confidence_label_audit_only_sample_count": sum(
            int(row["high_confidence_label_audit_only"]) for row in queue
        ),
        "partial_motion_only_sample_count": len(queue),
        "manual_review_required": True,
        "human_review_approved": False,
        "training_launch_allowed": False,
        "render_pass_grants_training_admission": False,
    }
    receipt["record_sha256"] = json_hash(receipt)
    atomic_json(output_root / "bundle_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
