#!/usr/bin/env python3
"""Validate and package the fixed HAA500 interaction-primitives review sample."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "deliverables/interactive_human_motion_v1/catalog/haa500"
OUTPUT_DIR = ROOT / "deliverables/interactive_human_motion_v1/samples/haa500_primitives"
SELECTED_IDS = (
    "applauding_3_clip1",
    "bowing_waist_12_clip1",
    "bowing_fullbody_4_clip1",
    "hailing_taxi_15_clip1",
    "salute_11_clip1",
    "high_five_13_clip1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_candidates() -> dict[str, dict]:
    records = {}
    for name in (
        "communicative_primitives_pending_review.jsonl",
        "partner_offers_needs_review.jsonl",
    ):
        with (CATALOG_DIR / name).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records[record["clip_id"]] = record
    missing = set(SELECTED_IDS) - records.keys()
    if missing:
        raise RuntimeError(f"selected clips missing from source manifests: {sorted(missing)}")
    return records


def _codec_check(path: Path) -> dict:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-i",
        str(path),
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    report = result.stderr
    h264 = "Video: h264" in report
    yuv420p = "yuv420p" in report
    if not h264 or not yuv420p:
        raise RuntimeError(f"unexpected video encoding for {path}: h264={h264}, yuv420p={yuv420p}")
    return {"codec": "h264", "pixel_format": "yuv420p", "decode_exit_code": result.returncode}


def _video_check(path: Path, expected_frames: int) -> tuple[dict, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"decoded {len(frames)} frames from {path}, expected {expected_frames}")

    stack = np.stack(frames)
    height, width = stack.shape[1:3]
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    foreground = gray > 8
    ys, xs = np.where(np.any(foreground, axis=0))
    if not len(xs) or not len(ys):
        raise RuntimeError(f"blank video: {path}")
    edge_margin_px = int(min(xs.min(), width - 1 - xs.max(), ys.min(), height - 1 - ys.max()))
    frame_deltas = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0)), axis=(1, 2))
    motion_mean = float(np.mean(frame_deltas)) if len(frame_deltas) else 0.0
    pixel_std_mean = float(np.mean(np.std(gray, axis=(1, 2))))
    if motion_mean <= 0.01:
        raise RuntimeError(f"video has no measurable motion: {path}")
    if pixel_std_mean <= 1.0:
        raise RuntimeError(f"video is visually blank: {path}")
    if edge_margin_px <= 0:
        raise RuntimeError(f"robot or scene foreground touches frame edge: {path}")

    return {
        "decoded_frames": len(frames),
        "width": width,
        "height": height,
        "fps": fps,
        "pixel_std_mean": pixel_std_mean,
        "interframe_motion_mean": motion_mean,
        "foreground_edge_margin_px": edge_margin_px,
        "nonblank": True,
        "has_motion": True,
        "full_frame_uncropped": True,
    }, frames


def _peak_frame_index(csv_path: Path) -> int:
    trajectory = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    # Ignore head joints when selecting the representative communicative body pose.
    displacement = np.linalg.norm(trajectory[:, :15] - trajectory[0, :15], axis=1)
    return int(np.argmax(displacement))


def _contact_sheet(items: list[tuple[dict, np.ndarray]], output: Path) -> None:
    tile_width, tile_height = 640, 360
    canvas = Image.new("RGB", (tile_width * 2, tile_height * 3), (8, 8, 8))
    for index, (record, bgr_frame) in enumerate(items):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tile = Image.fromarray(rgb).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, tile_height - 42, tile_width, tile_height), fill=(0, 0, 0))
        label = f"{record['clip_id']} | {record['communicative_intent']}"
        draw.text((12, tile_height - 30), label, fill=(255, 255, 255))
        canvas.paste(tile, ((index % 2) * tile_width, (index // 2) * tile_height))
    canvas.save(output, quality=92)


def main() -> None:
    candidates = _read_candidates()
    manifest = []
    contact_items = []
    for clip_id in SELECTED_IDS:
        candidate = candidates[clip_id]
        if not candidate["physical_qc"]["passed"]:
            raise RuntimeError(f"physical QC is not passed: {clip_id}")
        if candidate["semantic_review"]["accepted_for_training"]:
            raise RuntimeError(f"review sample unexpectedly admitted for training: {clip_id}")

        video_path = OUTPUT_DIR / f"{clip_id}.mp4"
        render_path = OUTPUT_DIR / f"{clip_id}.render.json"
        render = json.loads(render_path.read_text(encoding="utf-8"))
        expected_frames = int(candidate["trajectory"]["frames"])
        video_check, frames = _video_check(video_path, expected_frames)
        video_check.update(_codec_check(video_path))
        if render["output_contract"] != "ula_v2_18d_head_v1" or render["action_dim"] != 18:
            raise RuntimeError(f"wrong render contract: {clip_id}")

        peak_index = _peak_frame_index(Path(candidate["trajectory"]["path"]))
        contact_items.append((candidate, frames[peak_index]))
        is_partner_offer = candidate["context_dependency"].startswith("partner_")
        manifest.append(
            {
                "schema_version": "1.0.0",
                "clip_id": clip_id,
                "communicative_intent": candidate["communicative_intent"],
                "canonical_prompt_en": candidate["canonical_prompt_en"],
                "source_action": candidate["source_action"],
                "sample_role": "partner_offer_probe" if is_partner_offer else "single_actor_communicative_primitive",
                "context_dependency": candidate["context_dependency"],
                "partner_conditioning_required": is_partner_offer,
                "accepted_for_training": False,
                "manual_video_review_required": True,
                "training_exclusion_reason": (
                    "partner pose, geometry, contact, and response timing are absent"
                    if is_partner_offer
                    else "semantic video review is not complete"
                ),
                "high_dynamic": False,
                "robot_contract": candidate["robot_contract"],
                "source": candidate["source"],
                "trajectory": candidate["trajectory"],
                "physical_qc": candidate["physical_qc"],
                "render": {
                    "video": video_path.name,
                    "video_sha256": _sha256(video_path),
                    "summary": render_path.name,
                    "summary_sha256": _sha256(render_path),
                    "model_source": render["model_source"],
                    "camera_framing": render["camera_framing"],
                    "representative_frame": peak_index,
                    "checks": video_check,
                },
            }
        )

    manifest_path = OUTPUT_DIR / "sample_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in manifest),
        encoding="utf-8",
    )
    _contact_sheet(contact_items, OUTPUT_DIR / "contact_sheet.jpg")
    print(json.dumps({"samples": len(manifest), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
