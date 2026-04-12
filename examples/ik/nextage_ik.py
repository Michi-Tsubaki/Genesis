import numpy as np
import genesis.utils.geom as gu


rarm_link_list = [
    "RARM_JOINT0",
    "RARM_JOINT1",
    "RARM_JOINT2",
    "RARM_JOINT3",
    "RARM_JOINT4",
    "RARM_JOINT5",
]

larm_link_list = [
    "LARM_JOINT0",
    "LARM_JOINT1",
    "LARM_JOINT2",
    "LARM_JOINT3",
    "LARM_JOINT4",
    "LARM_JOINT5",
]

ARM_JOINT_NAME_MAP = {
    "right": rarm_link_list,
    "left": larm_link_list,
}

ARM_END_EFFECTOR_LINK_MAP = {
    "right": "RARM_JOINT5_Link",
    "left": "LARM_JOINT5_Link",
}

NEXTAGE_RESET_POSE_DEG = {
    "CHEST_JOINT0": 0.0,
    "RARM_JOINT0": -0.6,
    "RARM_JOINT1": 0.0,
    "RARM_JOINT2": -100.0,
    "RARM_JOINT3": 15.2,
    "RARM_JOINT4": 9.4,
    "RARM_JOINT5": 3.2,
    "LARM_JOINT0": 0.6,
    "LARM_JOINT1": 0.0,
    "LARM_JOINT2": -100.0,
    "LARM_JOINT3": -15.2,
    "LARM_JOINT4": 9.4,
    "LARM_JOINT5": -3.2,
    "HEAD_JOINT0": 0.0,
    "HEAD_JOINT1": 0.0,
}

NEXTAGE_IK_KP = np.array([1500, 1500, 1200, 800, 600, 400], dtype=np.float32)
NEXTAGE_IK_KV = np.array([150, 150, 120, 80, 60, 40], dtype=np.float32)
NEXTAGE_IK_FORCE_LOWER = np.array([-150, -200, -100, -100, -100, -100], dtype=np.float32)
NEXTAGE_IK_FORCE_UPPER = np.array([150, 200, 100, 100, 100, 100], dtype=np.float32)


def get_arm_joint_names(arm):
    arm_key = arm.lower()
    if arm_key not in ARM_JOINT_NAME_MAP:
        raise ValueError(f"Unknown arm '{arm}'. Use 'right' or 'left'.")
    return ARM_JOINT_NAME_MAP[arm_key]


def get_arm_ik_dofs(robot, arm):
    return np.array(
        [robot.get_joint(name).dofs_idx_local[0] for name in get_arm_joint_names(arm)],
        dtype=int,
    )


def get_arm_end_effector(robot, arm):
    arm_key = arm.lower()
    if arm_key not in ARM_END_EFFECTOR_LINK_MAP:
        raise ValueError(f"Unknown arm '{arm}'. Use 'right' or 'left'.")
    return robot.get_link(ARM_END_EFFECTOR_LINK_MAP[arm_key])


def get_arm_ik_setup(robot, arm):
    ik_dofs = get_arm_ik_dofs(robot, arm)
    return {
        "joint_names": list(get_arm_joint_names(arm)),
        "ik_dofs": ik_dofs,
        "end_effector": get_arm_end_effector(robot, arm),
    }


def get_other_dofs(robot, ik_dofs):
    all_dofs = np.arange(robot.n_dofs)
    return np.setdiff1d(all_dofs, np.asarray(ik_dofs, dtype=int))


def configure_nextage_dofs(robot, ik_dofs):
    robot.set_dofs_kp(np.full(robot.n_dofs, 1000.0, dtype=np.float32))
    robot.set_dofs_kv(np.full(robot.n_dofs, 100.0, dtype=np.float32))
    robot.set_dofs_force_range(
        np.full(robot.n_dofs, -500.0, dtype=np.float32),
        np.full(robot.n_dofs, 500.0, dtype=np.float32),
    )

    robot.set_dofs_kp(NEXTAGE_IK_KP, ik_dofs)
    robot.set_dofs_kv(NEXTAGE_IK_KV, ik_dofs)
    robot.set_dofs_force_range(NEXTAGE_IK_FORCE_LOWER, NEXTAGE_IK_FORCE_UPPER, ik_dofs)


def make_default_pose(robot):
    default_pose = robot.get_dofs_position().cpu().numpy().astype(np.float32)
    if default_pose.ndim > 1:
        default_pose = default_pose[0].copy()
    for joint_name, angle_deg in NEXTAGE_RESET_POSE_DEG.items():
        dof_idx = robot.get_joint(joint_name).dofs_idx_local[0]
        default_pose[dof_idx] = np.deg2rad(angle_deg)
    return default_pose


def make_quat_from_rpy_deg(roll=0.0, pitch=0.0, yaw=0.0):
    return gu.xyz_to_quat(np.deg2rad(np.array([roll, pitch, yaw], dtype=np.float32)))


def get_target_pos(cube, offset=(0.0, 0.0, 0.0)):
    return cube.get_pos().cpu().numpy().astype(np.float32) + np.array(offset, dtype=np.float32)


def pose_to_T(pos, quat):
    return gu.trans_quat_to_T(
        np.asarray(pos, dtype=np.float32),
        np.asarray(quat, dtype=np.float32),
    )


def to_numpy_f32(array):
    if hasattr(array, "cpu"):
        return array.cpu().numpy().astype(np.float32)
    return np.asarray(array, dtype=np.float32)


def solve_arm_ik(robot, end_effector, ik_dofs, target_pos, target_quat, init_qpos=None, **kwargs):
    if init_qpos is None:
        init_qpos = to_numpy_f32(robot.get_qpos())
    qpos_goal = robot.inverse_kinematics(
        link=end_effector,
        pos=np.asarray(target_pos, dtype=np.float32),
        quat=np.asarray(target_quat, dtype=np.float32),
        init_qpos=np.asarray(init_qpos, dtype=np.float32),
        dofs_idx_local=np.asarray(ik_dofs, dtype=int),
        **kwargs,
    )
    qpos_goal = to_numpy_f32(qpos_goal)
    return qpos_goal[..., ik_dofs].astype(np.float32)
