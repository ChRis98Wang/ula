#!/usr/bin/env python3
import argparse
import json
import os
import re
import struct
import tempfile
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_URDF_SOURCE = REPO_ROOT / "urdf_V2_20260514" / "urdf" / "xacro" / "robot_modify.urdf"
NEW_URDF_PACKAGE_ROOT = NEW_URDF_SOURCE.parents[2]
NEW_URDF_MESH_DIR = NEW_URDF_PACKAGE_ROOT / "TorsoArm_urdf" / "meshes"
DEFAULT_URDF = NEW_URDF_SOURCE.with_name("robot_modify_meshdir.urdf")
MAX_MUJOCO_STL_FACES = 200_000
MUJOCO_ASSET_DIR = NEW_URDF_PACKAGE_ROOT / ".mujoco_assets"


def _binary_stl_face_count(mesh_path):
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() != ".stl" or not mesh_path.is_file():
        return None
    header = mesh_path.read_bytes()[:84]
    if len(header) < 84:
        return None
    face_count = struct.unpack("<I", header[80:84])[0]
    expected_size = 84 + 50 * face_count
    if expected_size != mesh_path.stat().st_size:
        return None
    return int(face_count)


def _obj_from_binary_stl(mesh_path, package_root):
    mesh_path = Path(mesh_path)
    output = Path(package_root) / ".mujoco_assets" / f"{mesh_path.stem}.obj"
    if output.exists() and output.stat().st_mtime >= mesh_path.stat().st_mtime:
        return output
    face_count = _binary_stl_face_count(mesh_path)
    if face_count is None:
        return mesh_path
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with mesh_path.open("rb") as source, temporary.open("w", encoding="ascii") as target:
            source.seek(84)
            vertex_index = 1
            for _ in range(face_count):
                record = source.read(50)
                vertices = struct.unpack("<12fH", record)[3:12]
                for start in (0, 3, 6):
                    target.write(
                        f"v {vertices[start]:.9g} {vertices[start + 1]:.9g} "
                        f"{vertices[start + 2]:.9g}\n"
                    )
                target.write(f"f {vertex_index} {vertex_index + 1} {vertex_index + 2}\n")
                vertex_index += 3
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _mujoco_safe_mesh(mesh_path, package_root):
    face_count = _binary_stl_face_count(mesh_path)
    if face_count is not None and face_count > MAX_MUJOCO_STL_FACES:
        return _obj_from_binary_stl(mesh_path, package_root)
    return Path(mesh_path)


def ensure_mujoco_urdf(urdf_path=NEW_URDF_SOURCE, output_path=None):
    urdf_path = Path(urdf_path)
    if output_path is None:
        output_path = urdf_path.with_name(f"{urdf_path.stem}_meshdir.urdf")
    output_path = Path(output_path)
    package_root = urdf_path.parents[2]
    meshes_by_name = {}
    for mesh_path in package_root.rglob("*.STL"):
        meshes_by_name.setdefault(mesh_path.name, []).append(mesh_path)
    text = urdf_path.read_text(encoding="utf-8")

    def replace_mesh(match):
        filename = match.group(1)
        candidate = None
        if filename.startswith("package://"):
            relative = filename.removeprefix("package://")
            relative_parts = Path(relative).parts
            if relative_parts and relative_parts[0] == package_root.name:
                relative = str(Path(*relative_parts[1:]))
            candidate = package_root / relative
        elif Path(filename).is_absolute():
            candidate = Path(filename)
        mesh_name = Path(filename).name
        if candidate is None or not candidate.exists():
            candidates = meshes_by_name.get(mesh_name, [])
            if len(candidates) == 1:
                candidate = candidates[0]
        if candidate is not None and candidate.exists():
            return f'filename="{_mujoco_safe_mesh(candidate, package_root)}"'
        return f'filename="{filename}"'

    text = re.sub(r'filename="([^"]+\.(?:STL|stl))"', replace_mesh, text)
    if "<mujoco>" not in text:
        text = re.sub(
            r"(<robot\b[^>]*>)",
            '\\1\n  <mujoco>\n    <compiler fusestatic="false" strippath="false" />\n  </mujoco>',
            text,
            count=1,
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def resolve_mujoco_urdf(urdf_path=DEFAULT_URDF):
    urdf_path = Path(urdf_path)
    if urdf_path == DEFAULT_URDF:
        return ensure_mujoco_urdf(NEW_URDF_SOURCE, DEFAULT_URDF)
    if urdf_path.name == "robot_modify.urdf":
        return ensure_mujoco_urdf(urdf_path)
    return urdf_path

V2_ARMS_DOWN_SHOULDER_ROLL = -1.4
DELTA_RAD = 0.2


JOINT_TO_WRIST_BODY = {
    "joint_lShoulderPitch": "link_lWristPitch",
    "joint_lShoulderRoll": "link_lWristPitch",
    "joint_lShoulderYaw": "link_lWristPitch",
    "joint_lElbow": "link_lWristPitch",
    "joint_lWristRoll": "link_lWristPitch",
    "joint_lWristPitch": "link_lWristPitch",
    "joint_rShoulderPitch": "link_rWristPitch",
    "joint_rShoulderRoll": "link_rWristPitch",
    "joint_rShoulderYaw": "link_rWristPitch",
    "joint_rElbow": "link_rWristPitch",
    "joint_rWristRoll": "link_rWristPitch",
    "joint_rWristPitch": "link_rWristPitch",
}


def _name_maps(model):
    return (
        {model.joint(i).name: int(model.joint(i).qposadr[0]) for i in range(model.njnt)},
        {model.body(i).name: i for i in range(model.nbody)},
    )


def _body_xpos(model, data, qpos, body_name, body_ids):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.xpos[body_ids[body_name]].copy()


def analyze_v2_axes(urdf_path=DEFAULT_URDF, delta_rad=DELTA_RAD):
    urdf_path = resolve_mujoco_urdf(urdf_path)
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    qpos_addr, body_ids = _name_maps(model)

    base_qpos = np.zeros(model.nq)
    if "joint_lShoulderRoll" in qpos_addr:
        base_qpos[qpos_addr["joint_lShoulderRoll"]] = V2_ARMS_DOWN_SHOULDER_ROLL
    if "joint_rShoulderRoll" in qpos_addr:
        base_qpos[qpos_addr["joint_rShoulderRoll"]] = V2_ARMS_DOWN_SHOULDER_ROLL

    joint_report = {}
    for joint, wrist_body in JOINT_TO_WRIST_BODY.items():
        if joint not in qpos_addr or wrist_body not in body_ids:
            continue
        q_plus = base_qpos.copy()
        q_plus[qpos_addr[joint]] += delta_rad
        base = _body_xpos(model, data, base_qpos, wrist_body, body_ids)
        moved = _body_xpos(model, data, q_plus, wrist_body, body_ids)
        delta = moved - base
        joint_report[joint] = {
            "wrist_body": wrist_body,
            "delta_rad": float(delta_rad),
            "base_wrist_position": [float(v) for v in base],
            "moved_wrist_position": [float(v) for v in moved],
            "wrist_delta": [float(v) for v in delta],
            "delta_norm": float(np.linalg.norm(delta)),
        }

    return {
        "urdf_path": str(urdf_path),
        "base_pose": {
            "joint_lShoulderRoll": V2_ARMS_DOWN_SHOULDER_ROLL,
            "joint_rShoulderRoll": V2_ARMS_DOWN_SHOULDER_ROLL,
        },
        "joints": joint_report,
        "calibration": {
            "joint_lShoulderRoll": {
                "arms_down_offset_rad": V2_ARMS_DOWN_SHOULDER_ROLL,
                "human_abduction_sign": 1.0,
            },
            "joint_rShoulderRoll": {
                "arms_down_offset_rad": V2_ARMS_DOWN_SHOULDER_ROLL,
                "human_abduction_sign": 1.0,
            },
            "joint_lElbow": {
                "human_flexion_sign": -1.0,
                "reason": "0514 URDF positive left elbow moves wrist toward x-, away from human flexion branch",
            },
            "joint_rElbow": {
                "human_flexion_sign": -1.0,
                "reason": "0514 URDF positive right elbow also moves wrist toward x-, so right flexion sign matches left",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze V2 MuJoCo joint axis directions by small perturbations")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta-rad", type=float, default=DELTA_RAD)
    args = parser.parse_args()
    report = analyze_v2_axes(args.urdf, delta_rad=args.delta_rad)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "joint_count": len(report["joints"])}, indent=2))


if __name__ == "__main__":
    main()
