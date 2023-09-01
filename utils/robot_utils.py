import numpy as np
import sapien.core as sapien

from pathlib import Path
from typing import NamedTuple, List

class RobotInfo(NamedTuple):
    path: str
    arm_dof: int
    hand_dof: int
    palm_name: str
    arm_init_qpos: List[float]
    # root_offset: List[float] = [0.0, 0.0, 0.0]

def generate_robot_info():
    xarm6_path = Path("./assets/robot/xarm6_description/")
    xarm6_allegro_hand_left = RobotInfo(path=str(xarm6_path / "xarm6_allegro_long_finger_tip_left.urdf"), arm_dof=6, hand_dof=16,
                                        palm_name="palm_center", arm_init_qpos=[0, 0, 0, 0, -np.pi / 2, np.pi])
    xarm6_allegro_hand_right = RobotInfo(path=str(xarm6_path / "xarm6_allegro_long_finger_tip_right.urdf"), arm_dof=6, hand_dof=16, 
                                        palm_name="palm_center", arm_init_qpos=[0, 0, 0, 0, -np.pi / 2, 0])
    
    xarm7_path = Path("./assets/robot/xarm7_description/")
    xarm7_allegro_hand_left = RobotInfo(path=str(xarm7_path / "xarm7_allegro_long_finger_tip_left.urdf"), arm_dof=7, hand_dof=16, 
                                        palm_name="palm_center", arm_init_qpos=[0, 0, 0, 0, 0, -np.pi / 2, np.pi])
    xarm7_allegro_hand_right = RobotInfo(path=str(xarm7_path / "xarm7_allegro_long_finger_tip_right.urdf"), arm_dof=7, hand_dof=16,
                                        palm_name="palm_center", arm_init_qpos=[0, 0, 0, 0, 0, -np.pi / 2, 0])

    info_dict = dict(xarm6_allegro_hand_left=xarm6_allegro_hand_left, 
                     xarm6_allegro_hand_right=xarm6_allegro_hand_right,
                     xarm7_allegro_hand_left=xarm7_allegro_hand_left,
                     xarm7_allegro_hand_right=xarm7_allegro_hand_right)

    return info_dict

def load_robot(scene: sapien.Scene, robot_name: str, disable_self_collision: bool = False) -> sapien.Articulation:
    """
        Load robot, set up collsion, drive(control) and visual property.
    """
    # Load robot
    loader = scene.create_urdf_loader()
    info = generate_robot_info()[robot_name]
    filename = info.path
    robot_builder = loader.load_file_as_articulation_builder(filename)
    # Set up collision property
    if 'allegro' in robot_name:
        if disable_self_collision:
            for link_builder in robot_builder.get_link_builders():
                link_builder.set_collision_groups(1, 1, 17, 0)
        else:
            for link_builder in robot_builder.get_link_builders():
                # NOTE(chichu): These links are at the junction of palm and fingers
                if link_builder.get_name() in ["link_9.0", "link_5.0", "link_1.0", "link_13.0", "base_link"]:
                    link_builder.set_collision_groups(1, 1, 17, 0)
    elif 'ability' in robot_name:
        pass
    else:
        raise NotImplementedError
    robot = robot_builder.build(fix_root_link=True)
    robot.set_name(robot_name)
    # Set up drive(control) property
    arm_control_params = np.array([2e5, 4e4, 5e2])  # This PD is far larger than real to improve stability
    hand_control_params = np.array([2e2, 6e1, 1e1])
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]    # NOTE(chichu):This setting is compataible with both xarm6 and xarm7.
    for joint in robot.get_active_joints():
        name = joint.get_name()
        if name in arm_joint_names:
            joint.set_drive_property(*(1 * arm_control_params), mode="force")
        else:
            joint.set_drive_property(*(1 * hand_control_params), mode="force")
    # Set up visual material
    mat_physi = scene.engine.create_physical_material(1.5, 1, 0.01)
    for link in robot.get_links():
        for geom in link.get_collision_shapes():
            geom.min_patch_radius = 0.02
            geom.patch_radius = 0.04
            geom.set_physical_material(mat_physi)

    return robot