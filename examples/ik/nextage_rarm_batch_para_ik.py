import numpy as np
import genesis as gs

from nextage_ik import (
    get_arm_ik_setup,
    make_default_pose,
    make_quat_from_rpy_deg,
    solve_arm_ik,
    to_numpy_f32,
)


gs.init()

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.0, -2.8, 1.8),
        camera_lookat=(0.0, 0.0, 1.0),
        camera_fov=40,
        max_FPS=200,
    ),
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=False,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

robot = scene.add_entity(
    gs.morphs.MJCF(
        file="xml/nextage/nextage.xml",
        recompute_inertia=True,
        pos=(0.0, 0.0, 1.0),
    ),
    material=gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
)

n_envs = 16
scene.build(n_envs=n_envs, env_spacing=(1.4, 1.4))

arm_setup = get_arm_ik_setup(robot, "right")
ik_dofs = arm_setup["ik_dofs"]
ee_link = arm_setup["end_effector"]

default_pose = make_default_pose(robot)
robot.set_qpos(np.tile(default_pose, (n_envs, 1)))

target_quat = np.tile(make_quat_from_rpy_deg(0.0, -90.0, 0.0), (n_envs, 1))
center = np.tile(np.array([0.25, -0.10, 1.15], dtype=np.float32), (n_envs, 1))
angular_speed = np.random.uniform(-10.0, 10.0, n_envs).astype(np.float32)
radius = 0.08

for i in range(1000):
    target_pos = np.zeros((n_envs, 3), dtype=np.float32)
    target_pos[:, 0] = center[:, 0] + np.cos(i / 360.0 * np.pi * angular_speed) * radius
    target_pos[:, 1] = center[:, 1] + np.sin(i / 360.0 * np.pi * angular_speed) * radius
    target_pos[:, 2] = center[:, 2]

    q = solve_arm_ik(
        robot,
        ee_link,
        ik_dofs,
        target_pos,
        target_quat,
        init_qpos=to_numpy_f32(robot.get_qpos()),
    )

    current_q = to_numpy_f32(robot.get_qpos())
    current_q[:, ik_dofs] = q
    robot.set_qpos(current_q)
    scene.step()
