#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from upper_body_skeleton.extract import convert_npz_keypoints
from upper_body_skeleton.retarget_monitor import analyze_retarget_quality, write_frame_csv
from upper_body_skeleton.retarget_v2 import retarget_payload_to_rows, write_joint_csv
from upper_body_skeleton.retarget_v2_ik import retarget_payload_to_rows_ik
from upper_body_skeleton.smooth import smooth_payload


@dataclass(frozen=True)
class BatchOutputPaths:
    manifest_key: str
    work_dir: Path
    skeleton_json: Path
    joint_csv: Path
    retarget_report: Path
    monitor_json: Path
    monitor_csv: Path


def discover_npz_files(extracted_root):
    return sorted(Path(extracted_root).rglob("*.npz"))


def output_paths_for_npz(npz_path, extracted_root, output_root):
    npz_path = Path(npz_path)
    extracted_root = Path(extracted_root)
    output_root = Path(output_root)
    relative_stem = npz_path.relative_to(extracted_root).with_suffix("")
    work_dir = output_root / relative_stem
    stem = npz_path.stem
    return BatchOutputPaths(
        manifest_key=relative_stem.as_posix(),
        work_dir=work_dir,
        skeleton_json=work_dir / f"{stem}.keypoint_upper_body_skeleton_smoothed.json",
        joint_csv=work_dir / f"{stem}.v2_upper_body_joints.csv",
        retarget_report=work_dir / f"{stem}.v2_retarget_report.json",
        monitor_json=work_dir / f"{stem}.v2_monitor_report.json",
        monitor_csv=work_dir / f"{stem}.v2_monitor_frames.csv",
    )


def resolve_max_frames(max_frames, full_video=False):
    return None if full_video else max_frames


def resolve_retarget_mode(mode):
    if mode not in {"formula", "ik"}:
        raise ValueError("retarget_mode must be 'formula' or 'ik'")
    return mode


def select_shard(files, shard_index=0, shard_count=1):
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return [path for index, path in enumerate(files) if index % shard_count == shard_index]


def should_reuse_skeleton(skeleton_json, overwrite=False):
    return Path(skeleton_json).exists() and not overwrite


def should_write_progress_manifest(index, total, interval=25):
    if index == 1 or index == total:
        return True
    return interval > 0 and index % interval == 0


def process_npz(
    npz_path,
    extracted_root,
    output_root,
    fps=30.0,
    max_frames=180,
    stride=1,
    output_hz=30.0,
    smooth_window_frames=11,
    image_size=None,
    overwrite=False,
    start_frame=0,
    retarget_mode="formula",
):
    paths = output_paths_for_npz(npz_path, extracted_root, output_root)
    video_path = Path(npz_path).with_suffix(".mp4")
    if paths.monitor_json.exists() and not overwrite:
        monitor = json.loads(paths.monitor_json.read_text(encoding="utf-8"))
        return {
            "sample": paths.manifest_key,
            "npz_path": str(npz_path),
            "video_path": str(video_path) if video_path.exists() else "",
            "status": "skipped_existing",
            "frame_count": monitor.get("summary", {}).get("frame_count", 0),
            "flagged_frame_count": monitor.get("summary", {}).get("flagged_frame_count", 0),
            "max_cross_body_intent": monitor.get("summary", {}).get("max_cross_body_intent", 0.0),
            "max_yaw_under_response": monitor.get("summary", {}).get("max_yaw_under_response", 0.0),
            "max_elbow_overfold": monitor.get("summary", {}).get("max_elbow_overfold", 0.0),
            "skeleton_json": str(paths.skeleton_json),
            "joint_csv": str(paths.joint_csv),
            "monitor_json": str(paths.monitor_json),
        }

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    if should_reuse_skeleton(paths.skeleton_json, overwrite=overwrite):
        smoothed = json.loads(paths.skeleton_json.read_text(encoding="utf-8"))
    else:
        raw_probe_path = paths.work_dir / f"{Path(npz_path).stem}.keypoint_upper_body_skeleton_raw.tmp.json"
        payload = convert_npz_keypoints(
            npz_path,
            raw_probe_path,
            fps=fps,
            max_frames=max_frames,
            stride=stride,
            image_size=image_size or [1080.0, 1920.0],
            start_frame=start_frame,
        )
        if raw_probe_path.exists():
            raw_probe_path.unlink()

        smoothed = smooth_payload(payload, window_frames=smooth_window_frames)
        paths.skeleton_json.write_text(json.dumps(smoothed, indent=2), encoding="utf-8")

    retarget_mode = resolve_retarget_mode(retarget_mode)
    if retarget_mode == "ik":
        rows = retarget_payload_to_rows_ik(smoothed)
        report = {
            "source_frame_count": len(smoothed.get("frames", [])),
            "row_count": len(rows),
            "output_hz": float(output_hz),
            "retarget_mode": "ik",
            "joint_order": rows and list(rows[0].keys()) or [],
            "notes": [
                "MuJoCo FK-guided IK uses V2-coordinate video-plane upper-body targets.",
                "Lower body and balance are intentionally excluded.",
            ],
        }
    else:
        rows, report = retarget_payload_to_rows(smoothed, output_hz=output_hz)
        report["retarget_mode"] = "formula"
    write_joint_csv(rows, paths.joint_csv)
    paths.retarget_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    monitor = analyze_retarget_quality(smoothed, paths.joint_csv)
    paths.monitor_json.write_text(json.dumps(monitor, indent=2), encoding="utf-8")
    write_frame_csv(monitor, paths.monitor_csv)
    summary = monitor["summary"]
    return {
        "sample": paths.manifest_key,
        "npz_path": str(npz_path),
        "video_path": str(video_path) if video_path.exists() else "",
        "status": "processed",
        "frame_count": summary["frame_count"],
        "flagged_frame_count": summary["flagged_frame_count"],
        "max_cross_body_intent": summary["max_cross_body_intent"],
        "max_yaw_under_response": summary["max_yaw_under_response"],
        "max_elbow_overfold": summary["max_elbow_overfold"],
        "skeleton_json": str(paths.skeleton_json),
        "joint_csv": str(paths.joint_csv),
        "monitor_json": str(paths.monitor_json),
    }


def write_manifest(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "npz_path",
        "video_path",
        "status",
        "frame_count",
        "flagged_frame_count",
        "max_cross_body_intent",
        "max_yaw_under_response",
        "max_elbow_overfold",
        "skeleton_json",
        "joint_csv",
        "monitor_json",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Batch retarget Seamless keypoint NPZ files to V2 upper-body joints")
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", help="Defaults to <output-root>/manifest.csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--full-video", action="store_true", help="Process every frame instead of the max-frames slice")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-hz", type=float, default=30.0)
    parser.add_argument("--retarget-mode", choices=["formula", "ik"], default="formula")
    parser.add_argument("--smooth-window-frames", type=int, default=11)
    parser.add_argument("--image-width", type=float, default=1080.0)
    parser.add_argument("--image-height", type=float, default=1920.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    extracted_root = Path(args.extracted_root)
    output_root = Path(args.output_root)
    npz_files = discover_npz_files(extracted_root)
    npz_files = select_shard(npz_files, shard_index=args.shard_index, shard_count=args.shard_count)
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    manifest_rows = []
    manifest = Path(args.manifest) if args.manifest else output_root / "manifest.csv"
    for index, npz_path in enumerate(npz_files, start=1):
        try:
            row = process_npz(
                npz_path=npz_path,
                extracted_root=extracted_root,
                output_root=output_root,
                fps=args.fps,
                max_frames=resolve_max_frames(args.max_frames, args.full_video),
                stride=args.stride,
                output_hz=args.output_hz,
                smooth_window_frames=args.smooth_window_frames,
                image_size=[args.image_width, args.image_height],
                overwrite=args.overwrite,
                start_frame=args.start_frame,
                retarget_mode=args.retarget_mode,
            )
        except Exception as exc:
            row = {
                "sample": Path(npz_path).stem,
                "npz_path": str(npz_path),
                "video_path": str(Path(npz_path).with_suffix(".mp4")),
                "status": f"error:{type(exc).__name__}:{exc}",
                "frame_count": 0,
                "flagged_frame_count": 0,
                "max_cross_body_intent": 0.0,
                "max_yaw_under_response": 0.0,
                "max_elbow_overfold": 0.0,
                "skeleton_json": "",
                "joint_csv": "",
                "monitor_json": "",
            }
        manifest_rows.append(row)
        if should_write_progress_manifest(index, len(npz_files), interval=args.progress_interval):
            write_manifest(manifest_rows, manifest)
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(npz_files),
                        "sample": row["sample"],
                        "status": row["status"],
                        "manifest": str(manifest),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_manifest(manifest_rows, manifest)
    print(json.dumps({"samples": len(manifest_rows), "manifest": str(manifest)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
