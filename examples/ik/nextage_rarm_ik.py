import numpy as np
import genesis as gs

from nextage_ik import (
    configure_nextage_dofs,
    get_arm_ik_setup,
    get_other_dofs,
    get_target_pos,
    make_default_pose,
    make_quat_from_rpy_deg,
    pose_to_T,
    solve_arm_ik,
    to_numpy_f32,
)


gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.6, -1.6, 1.8),
        camera_lookat=(0.0, 0.0, 1.15),
        camera_fov=35,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=0.5,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(
        pos=(0.0, 0.0, 0.0),
    ),
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

target_cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.40, 0.0, 1.02),
    ),
    surface=gs.surfaces.Default(
        color=(1.0, 0.0, 0.0, 1.0),
    ),
    material=gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
)

desk = scene.add_entity(
    gs.morphs.Box(
        size=(0.5, 0.5, 1.00),
        pos=(0.4, 0.0, 0.50),
        fixed=True,
    ),
    material=gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
)

scene.build()

arm_setup = get_arm_ik_setup(robot, "right")
ik_dofs = arm_setup["ik_dofs"]
end_effector = arm_setup["end_effector"]
other_dofs = get_other_dofs(robot, ik_dofs)

configure_nextage_dofs(robot, ik_dofs)
default_pose = make_default_pose(robot)

ee_frame = scene.draw_debug_frame(
    pose_to_T(to_numpy_f32(end_effector.get_pos()), to_numpy_f32(end_effector.get_quat())),
    axis_length=0.12,
    origin_size=0.012,
    axis_radius=0.006,
    color=(0.3, 0.7, 1.0, 1.0),
)

target_frame = scene.draw_debug_frame(
    pose_to_T(get_target_pos(target_cube), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
    axis_length=0.12,
    origin_size=0.012,
    axis_radius=0.006,
    color=(1.0, 0.4, 0.4, 1.0),
)


def update_debug_frames(target_pos, target_quat):
    ee_T = pose_to_T(to_numpy_f32(end_effector.get_pos()), to_numpy_f32(end_effector.get_quat()))
    target_T = pose_to_T(target_pos, target_quat)
    scene.update_debug_objects((ee_frame, target_frame), (ee_T, target_T))


def solve_ik(target_pos, target_quat):
    return solve_arm_ik(robot, end_effector, ik_dofs, target_pos, target_quat)


def angle_vector_sequence(goal, target_pos, target_quat, steps):
    for _ in range(steps):
        robot.control_dofs_position(goal, ik_dofs)
        robot.control_dofs_position(default_pose[other_dofs], other_dofs)
        update_debug_frames(target_pos, target_quat)
        scene.step()


def motion_planning(target_pos, target_quat, steps):
    start = to_numpy_f32(robot.get_dofs_position(ik_dofs))
    goal = solve_ik(target_pos, target_quat)
    for a in np.linspace(0.0, 1.0, steps, dtype=np.float32):
        cmd = ((1.0 - a) * start + a * goal).astype(np.float32)
        robot.control_dofs_position(cmd, ik_dofs)
        robot.control_dofs_position(default_pose[other_dofs], other_dofs)
        update_debug_frames(target_pos, target_quat)
        scene.step()
    return goal


robot.set_dofs_position(default_pose, zero_velocity=True)
for i in range(10):
    robot.control_dofs_position(default_pose)
    update_debug_frames(get_target_pos(target_cube), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    scene.step()
    print(i, "base_pos =", to_numpy_f32(robot.get_pos()), "base_quat =", to_numpy_f32(robot.get_quat()))

ee_pos = to_numpy_f32(end_effector.get_pos())
target_pos = get_target_pos(target_cube)
clearance_z = max(ee_pos[2] + 0.12, target_pos[2] + 0.20)

lift_pos = np.array([ee_pos[0] + 0.20, ee_pos[1], clearance_z], dtype=np.float32)
travel_quat = make_quat_from_rpy_deg(0.0, -90.0, 0.0)
goal = motion_planning(lift_pos, travel_quat, 180)
angle_vector_sequence(goal, lift_pos, travel_quat, 60)

hover_pos = np.array([target_pos[0], target_pos[1], clearance_z], dtype=np.float32)
travel_quat = make_quat_from_rpy_deg(0.0, -90.0, 0.0)
goal = motion_planning(hover_pos, travel_quat, 220)
angle_vector_sequence(goal, hover_pos, travel_quat, 60)

approach_pos = get_target_pos(target_cube, (0.0, 0.0, 0.00))
travel_quat = make_quat_from_rpy_deg(0.0, -90.0, 0.0)
goal = motion_planning(approach_pos, travel_quat, 180)
angle_vector_sequence(goal, approach_pos, travel_quat, 120)

retreat_pos = get_target_pos(target_cube, (0.0, 0.0, 0.20))
retreat_quat = make_quat_from_rpy_deg(0.0, -90.0, 0.0)
goal = motion_planning(retreat_pos, retreat_quat, 180)
angle_vector_sequence(goal, retreat_pos, retreat_quat, 180)
