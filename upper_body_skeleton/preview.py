#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


EDGES = [
    ("pelvis_origin", "torso_origin"),
    ("torso_origin", "neck"),
    ("neck", "head"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]


def plot_preview(skeleton_json, output_png, frame_indices):
    with open(skeleton_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]
    selected = [frames[i] for i in frame_indices if 0 <= i < len(frames)]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    all_points = []
    for idx, frame in enumerate(selected):
        lm = frame["landmarks_3d"]
        color = colors[idx % len(colors)]
        for a, b in EDGES:
            xs = [lm[a][0], lm[b][0]]
            ys = [lm[a][1], lm[b][1]]
            zs = [lm[a][2], lm[b][2]]
            ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=0.85)
            all_points.extend([lm[a], lm[b]])
        ax.text(lm["head"][0], lm["head"][1], lm["head"][2], f"f{frame['frame_index']}", color=color)
    if all_points:
        xs, ys, zs = zip(*all_points)
        center = [(min(vals) + max(vals)) / 2 for vals in (xs, ys, zs)]
        radius = max(max(vals) - min(vals) for vals in (xs, ys, zs)) / 2
        radius = max(radius, 0.3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x forward")
    ax.set_ylabel("y left")
    ax.set_zlabel("z up")
    ax.set_title("URDF-zero upper-body skeleton preview")
    ax.view_init(elev=18, azim=-70)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render upper-body skeleton preview")
    parser.add_argument("skeleton_json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", default="0,45,90,135,179")
    args = parser.parse_args()
    frame_indices = [int(x) for x in args.frames.split(",") if x.strip()]
    plot_preview(args.skeleton_json, args.output, frame_indices)
    print(args.output)


if __name__ == "__main__":
    main()
