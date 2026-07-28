#!/usr/bin/env python3
"""Render representative Hanyang clips rejected by the strict robot QC gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from tools.human_motion_review.build_hanyang_training_sample_review import (
    DEFAULT_RENDERER_PYTHON,
    DEFAULT_RESEARCH_ROOT,
    DEFAULT_URDF,
    HEIGHT,
    OUTPUT_WIDTH,
    PANEL_WIDTH,
    ROBOT_WIDTH,
    SOURCE_FRAMES,
    SOURCE_FPS,
    SOURCE_WIDTH,
    _font,
    _stream_duration,
    assert_no_forbidden_data_lineage,
    atomic_json,
    atomic_jsonl,
    compose_segment,
    concat_segments,
    encode_frames,
    json_hash,
    load_hanyang_csv,
    render_robot,
    sha256_file,
    source_skeleton_frames,
)


DEFAULT_FAILED_MANIFEST = (
    DEFAULT_RESEARCH_ROOT / "retarget_v1" / "failed_manifest.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_RESEARCH_ROOT / "human_review" / "rejected_sample_v1"
)
EXPECTED_FAILED_MANIFEST_SHA256 = (
    "dd826f2ca5ee3c1c4700c4e795770988e9f3fb0537670d35d0ffd10858b271ac"
)
ARTIFACT_KIND = "hanyang_strict_qc_rejected_sample_review_bundle_v1"
QUEUE_KIND = "hanyang_strict_qc_rejected_sample_review_item_v1"

# Each emotion has a near-boundary/single-gate example and a distinct,
# deliberately severe multi-gate example.  These are diagnostic rejects, not
# candidates that can be approved into training from this review artifact.
SELECTION = (
    ("10_3_5_1", "近阈值：顶关节", ("saturation",)),
    (
        "29_2_5_1",
        "严重：全部 7 项",
        (
            "collision",
            "head_tilt_proxy",
            "limb_direction",
            "per_joint_velocity",
            "saturation",
            "source_geometry",
            "target_fit",
        ),
    ),
    ("17_2_3_2", "单项失败：碰撞", ("collision",)),
    (
        "1_2_2_2",
        "严重：全部 7 项",
        (
            "collision",
            "head_tilt_proxy",
            "limb_direction",
            "per_joint_velocity",
            "saturation",
            "source_geometry",
            "target_fit",
        ),
    ),
    ("23_4_4_3", "近阈值：源几何", ("source_geometry",)),
    (
        "21_4_4_3",
        "严重：全部 7 项",
        (
            "collision",
            "head_tilt_proxy",
            "limb_direction",
            "per_joint_velocity",
            "saturation",
            "source_geometry",
            "target_fit",
        ),
    ),
    ("16_4_5_4", "单项临界：头部", ("head_tilt_proxy",)),
    (
        "28_3_2_4",
        "严重：全部 7 项",
        (
            "collision",
            "head_tilt_proxy",
            "limb_direction",
            "per_joint_velocity",
            "saturation",
            "source_geometry",
            "target_fit",
        ),
    ),
    ("11_1_3_5", "近阈值：速度", ("per_joint_velocity",)),
    (
        "25_4_4_5",
        "严重：全部 7 项",
        (
            "collision",
            "head_tilt_proxy",
            "limb_direction",
            "per_joint_velocity",
            "saturation",
            "source_geometry",
            "target_fit",
        ),
    ),
    ("13_4_4_6", "近阈值：速度", ("per_joint_velocity",)),
    (
        "14_2_3_6",
        "严重：全部 7 项",
        (
            "collision",
            "head_tilt_proxy",
            "limb_direction",
            "per_joint_velocity",
            "saturation",
            "source_geometry",
            "target_fit",
        ),
    ),
    (
        "2_3_5_7",
        "双项临界：方向+拟合",
        ("limb_direction", "target_fit"),
    ),
    (
        "22_3_4_7",
        "严重：6 项失败",
        (
            "collision",
            "limb_direction",
            "per_joint_velocity",
            "saturation",
            "source_geometry",
            "target_fit",
        ),
    ),
)

GATE_LABELS = {
    "collision": "碰撞",
    "head_tilt_proxy": "头部代理",
    "limb_direction": "手臂方向",
    "per_joint_velocity": "速度",
    "saturation": "顶关节",
    "source_geometry": "源几何",
    "target_fit": "目标拟合",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failed-manifest", type=Path, default=DEFAULT_FAILED_MANIFEST
    )
    parser.add_argument(
        "--expected-failed-manifest-sha256",
        default=EXPECTED_FAILED_MANIFEST_SHA256,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--renderer-python", type=Path, default=DEFAULT_RENDERER_PYTHON
    )
    return parser.parse_args()


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


def load_selected(
    rows: list[dict[str, Any]], *, retarget_root: Path
) -> list[dict[str, Any]]:
    indexed = {str(row["source_stem"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    for stem, role, expected_failed_gates in SELECTION:
        row = indexed.get(stem)
        if row is None:
            raise ValueError(f"frozen rejected sample is missing: {stem}")
        row_payload = {key: value for key, value in row.items() if key != "record_sha256"}
        quality_path = Path(str(row["quality_json"])).resolve()
        try:
            quality_path.relative_to(retarget_root)
        except ValueError as error:
            raise ValueError(f"{stem}: quality path escapes retarget root") from error
        if (
            row.get("record_sha256") != json_hash(row_payload)
            or row.get("status") != "failed_quality"
            or row.get("fixed_split_assignment")
            not in {"train", "validation", "test"}
            or row.get("kimodo_accessed_or_used") is not False
            or row.get("generator_foundation_eligible") is not False
            or not quality_path.is_file()
            or sha256_file(quality_path) != row.get("quality_json_sha256")
        ):
            raise ValueError(f"{stem}: rejected manifest contract mismatch")
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality_payload = {
            key: value for key, value in quality.items() if key != "record_sha256"
        }
        if (
            quality.get("record_sha256") != json_hash(quality_payload)
            or quality.get("record_sha256") != row.get("quality_record_sha256")
            or quality.get("clip_id") != row.get("clip_id")
            or (quality.get("quality_gate") or {}).get("passed") is not False
        ):
            raise ValueError(f"{stem}: quality report contract mismatch")
        failed_gates = tuple(
            key.removesuffix("_pass")
            for key, value in quality["quality_gate"].items()
            if key != "passed" and value is False
        )
        if frozenset(failed_gates) != frozenset(expected_failed_gates):
            raise ValueError(f"{stem}: frozen failed-gate set changed")
        selected.append(
            {
                "row": row,
                "quality": quality,
                "role": role,
                "failed_gates": failed_gates,
            }
        )
    if len({item["row"]["participant_id"] for item in selected}) != 14:
        raise ValueError("rejected review participant diversity regressed")
    return selected


def panel_image(sample: Mapping[str, Any], *, index: int) -> Image.Image:
    row = sample["row"]
    quality = sample["quality"]
    trajectory = quality["trajectory"]
    gate = quality["quality_gate"]
    failed = list(sample["failed_gates"])
    image = Image.new("RGB", (PANEL_WIDTH, HEIGHT), (247, 247, 248))
    draw = ImageDraw.Draw(image)
    title_font = _font(26)
    header_font = _font(21)
    body_font = _font(17)
    small_font = _font(15)
    red = (174, 47, 39)
    green = (30, 126, 82)
    grey = (85, 98, 113)

    draw.rectangle((0, 0, PANEL_WIDTH, 72), fill=(45, 36, 42))
    draw.text(
        (20, 16),
        f"严格筛选淘汰样本  {index:02d}/14",
        fill=(255, 250, 250),
        font=title_font,
    )
    y = 88
    draw.text((20, y), str(row["clip_id"]), fill=(20, 91, 126), font=header_font)
    y += 36

    def line(label: str, value: str, *, ok: bool | None = None) -> None:
        nonlocal y
        color = grey if ok is None else (green if ok else red)
        draw.text((20, y), label, fill=grey, font=small_font)
        draw.text((184, y - 2), value, fill=color, font=body_font)
        y += 29

    line("intended 情绪", str(row["emotion_id"]).upper())
    line("抽样角色", str(sample["role"]))
    line(
        "split / participant",
        f"{row['fixed_split_assignment']} / P{int(row['participant_id']):02d}",
    )
    y += 3
    draw.line((18, y, PANEL_WIDTH - 18, y), fill=(197, 202, 209), width=2)
    y += 12
    draw.text((20, y), f"失败项：{len(failed)}", fill=red, font=header_font)
    y += 32
    failed_labels = [GATE_LABELS[name] for name in failed]
    for start in range(0, len(failed_labels), 4):
        draw.text(
            (22, y),
            "、".join(failed_labels[start : start + 4]),
            fill=red,
            font=body_font,
        )
        y += 27
    y += 4

    velocity_ratio = float(trajectory["max_velocity_limit_ratio"])
    saturation = float(
        trajectory["observed_joint_max_saturation_fraction"]
    )
    line("速度比（≤1.0）", f"{velocity_ratio:.3f}", ok=gate["per_joint_velocity_pass"])
    line("顶关节比例（≤1%）", f"{100 * saturation:.1f}%", ok=gate["saturation_pass"])
    line(
        "目标误差 p95（≤40mm）",
        f"{1000 * float(quality['limb_target_error_p95_m']):.1f} mm",
        ok=gate["target_fit_pass"],
    )
    line(
        "方向 p95（≤15°）",
        f"{float(quality['limb_direction_error_p95_deg']):.1f}°",
        ok=gate["limb_direction_pass"],
    )
    line(
        "碰撞帧率（≤5%）",
        f"{100 * float(quality['upper_body_collision_frame_rate']):.1f}%",
        ok=gate["collision_pass"],
    )
    line(
        "头部误差 p95（≤5°）",
        f"{float(quality['head_tilt_proxy_error_p95_deg']):.1f}°",
        ok=gate["head_tilt_proxy_pass"],
    )
    line(
        "源几何",
        "PASS" if gate["source_geometry_pass"] else "FAIL",
        ok=gate["source_geometry_pass"],
    )

    y += 3
    draw.rounded_rectangle(
        (18, y, PANEL_WIDTH - 18, y + 70),
        radius=9,
        fill=(255, 235, 232),
        outline=(210, 116, 105),
        width=2,
    )
    draw.text(
        (30, y + 10),
        "仅用于解释筛选原因；",
        fill=(121, 52, 45),
        font=small_font,
    )
    draw.text(
        (30, y + 37),
        "不能从此视频直接批准进入训练。",
        fill=(121, 52, 45),
        font=body_font,
    )
    draw.rectangle((0, HEIGHT - 54, PANEL_WIDTH, HEIGHT), fill=(145, 42, 35))
    draw.text(
        (PANEL_WIDTH // 2, HEIGHT - 28),
        "已淘汰 / REJECTED",
        fill=(255, 255, 255),
        font=header_font,
        anchor="mm",
    )
    return image


def main() -> int:
    args = parse_args()
    manifest = args.failed_manifest.resolve()
    output_root = args.output_root.resolve()
    urdf = args.urdf.resolve()
    renderer_python = args.renderer_python.resolve()
    assert_no_forbidden_data_lineage(
        {
            "failed_manifest": str(manifest),
            "output_root": str(output_root),
            "urdf": str(urdf),
        },
        context="hanyang_rejected_sample_review",
    )
    if (
        not manifest.is_file()
        or sha256_file(manifest) != args.expected_failed_manifest_sha256
    ):
        raise ValueError("frozen Hanyang failed manifest hash changed")
    if not urdf.is_file() or not renderer_python.is_file():
        raise FileNotFoundError("renderer Python or URDF is missing")

    selected = load_selected(read_jsonl(manifest), retarget_root=manifest.parent)
    queue: list[dict[str, Any]] = []
    segments: list[Path] = []
    for index, sample in enumerate(selected, 1):
        row = sample["row"]
        quality = sample["quality"]
        stem = str(row["source_stem"])
        item_root = output_root / "items" / f"{index:02d}_{stem}"
        source_video = item_root / "source_19point.mp4"
        robot_video = item_root / "robot_failed_source_faithful_partial18d.mp4"
        robot_summary = item_root / "robot_render_summary.json"
        panel_path = item_root / "rejection_panel.png"
        segment = output_root / "segments" / f"{index:02d}_{stem}.mp4"

        source = load_hanyang_csv(quality["source_csv"])
        if source["source_sha256"] != quality["source_sha256"]:
            raise ValueError(f"{stem}: source CSV hash mismatch")
        faithful = Path(
            quality["outputs"]["source_faithful_partial_18d_csv"]
        ).resolve()
        faithful_sha = quality["outputs"][
            "source_faithful_partial_18d_csv_sha256"
        ]
        if not faithful.is_file() or sha256_file(faithful) != faithful_sha:
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
        panel = panel_image(sample, index=index)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_path)
        compose_segment(
            source_video, robot_video, panel_path, output=segment
        )
        segments.append(segment)

        queue_row = {
            "schema_version": "1.0.0",
            "artifact_kind": QUEUE_KIND,
            "review_index": index,
            "clip_id": row["clip_id"],
            "source_stem": stem,
            "participant_id": row["participant_id"],
            "fixed_split_assignment": row["fixed_split_assignment"],
            "intended_emotion": row["emotion_id"],
            "selection_role": sample["role"],
            "failed_quality_gates": list(sample["failed_gates"]),
            "strict_training_eligible": False,
            "review_cannot_grant_training_admission": True,
            "source_video": str(source_video),
            "source_video_sha256": sha256_file(source_video),
            "robot_video": str(robot_video),
            "robot_video_sha256": sha256_file(robot_video),
            "review_segment": str(segment),
            "review_segment_sha256": sha256_file(segment),
            "quality_json": row["quality_json"],
            "quality_json_sha256": row["quality_json_sha256"],
        }
        queue_row["record_sha256"] = json_hash(queue_row)
        queue.append(queue_row)

    queue_path = output_root / "review_queue.jsonl"
    atomic_jsonl(queue_path, queue)
    reel = output_root / "hanyang_rejected_sample_review_70s.mp4"
    concat_segments(segments, output=reel, root=output_root)
    duration = _stream_duration(reel)
    if not 69.5 <= duration <= 70.5:
        raise ValueError(f"rejected sample reel duration changed: {duration}")
    receipt = {
        "schema_version": "1.0.0",
        "artifact_kind": ARTIFACT_KIND,
        "created_utc": utc_now(),
        "failed_manifest": str(manifest),
        "failed_manifest_sha256": args.expected_failed_manifest_sha256,
        "sample_count": len(queue),
        "sample_policy": (
            "seven_emotions_each_one_diagnostic_boundary_reject_and_one_"
            "distinct_severe_reject_across_train_validation_test_v2"
        ),
        "reel": str(reel),
        "reel_sha256": sha256_file(reel),
        "reel_duration_sec": duration,
        "reel_frames": len(queue) * SOURCE_FRAMES,
        "width": OUTPUT_WIDTH,
        "height": HEIGHT,
        "review_queue": str(queue_path),
        "review_queue_sha256": sha256_file(queue_path),
        "strict_training_eligible_sample_count": 0,
        "review_cannot_grant_training_admission": True,
        "kimodo_accessed_or_used": False,
    }
    receipt["record_sha256"] = json_hash(receipt)
    atomic_json(output_root / "bundle_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
