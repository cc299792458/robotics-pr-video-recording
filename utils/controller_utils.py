import numpy as np
import sapien.core as sapien

from typing import Dict, Sequence

from agents.controllers import DictController
from agents.configs.allegro_hand import XArmAllegroDefaultConfig

def get_active_joint_indices(
    articulation: sapien.Articulation, joint_names: Sequence[str]
):
    all_joint_names = [x.name for x in articulation.get_active_joints()]
    joint_indices = [all_joint_names.index(x) for x in joint_names]
    return joint_indices

def get_active_joints(articulation: sapien.Articulation, joint_names: Sequence[str]):
    joints = articulation.get_active_joints()
    joint_indices = get_active_joint_indices(articulation, joint_names)
    return [joints[idx] for idx in joint_indices]

def set_up_controller(arm_name='xarm6', hand_name='allegro', control_mode=None, robot=None, **kwargs):
    if hand_name == 'allegro':
        config = XArmAllegroDefaultConfig(arm_name)
    elif hand_name == 'ability':
        raise NotImplementedError
    
    controller = DictController(config=config, control_mode=control_mode, robot=robot, **kwargs)
    return controller