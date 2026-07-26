"""Append-only ULA V2 contract with three head-orientation joints."""

from upper_body_skeleton.retarget_v2 import JOINT_LIMITS, JOINT_ORDER


CONTRACT_VERSION = "ula_v2_18d_head_v1"
HEAD_JOINT_ORDER = [
    "head_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
]
HEAD_JOINT_LIMITS = {
    "head_roll_joint": (-0.785, 0.785),
    "head_pitch_joint": (-1.57, 1.57),
    "head_yaw_joint": (-1.57, 1.57),
}
HEAD_JOINT_AXES = {
    "head_roll_joint": (1.0, 0.0, 0.0),
    "head_pitch_joint": (0.0, 1.0, 0.0),
    "head_yaw_joint": (0.0, 0.0, -1.0),
}
HEAD_JOINT_VELOCITY_LIMITS = {name: 12.0 for name in HEAD_JOINT_ORDER}

JOINT_ORDER_18D = [*JOINT_ORDER, *HEAD_JOINT_ORDER]
JOINT_LIMITS_18D = {**JOINT_LIMITS, **HEAD_JOINT_LIMITS}


def joint_order_for_action_dim(action_dim):
    action_dim = int(action_dim)
    if action_dim == len(JOINT_ORDER):
        return list(JOINT_ORDER)
    if action_dim == len(JOINT_ORDER_18D):
        return list(JOINT_ORDER_18D)
    raise ValueError(
        f"ULA V2 trajectory must have {len(JOINT_ORDER)} or "
        f"{len(JOINT_ORDER_18D)} joints, got {action_dim}"
    )


__all__ = [
    "CONTRACT_VERSION",
    "HEAD_JOINT_AXES",
    "HEAD_JOINT_LIMITS",
    "HEAD_JOINT_ORDER",
    "HEAD_JOINT_VELOCITY_LIMITS",
    "JOINT_LIMITS_18D",
    "JOINT_ORDER_18D",
    "joint_order_for_action_dim",
]
