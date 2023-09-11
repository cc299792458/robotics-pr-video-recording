import sapien.core as sapien

from agents.controllers import DictController
from agents.configs.allegro_hand import XArmAllegroDefaultConfig

def set_up_controller(arm_name='xarm6', hand_name='allegro', control_mode=None, robot=None, **kwargs):
    # if hand_name == 'allegro':
    #     config = XArmAllegroDefaultConfig(arm_name, hand_name)
    # elif hand_name == 'ability':
    #     
    config = XArmAllegroDefaultConfig(arm_name, hand_name)
    
    controller = DictController(config=config, control_mode=control_mode, robot=robot, **kwargs)
    return controller