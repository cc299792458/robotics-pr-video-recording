import numpy as np

from typing import Union, Sequence
from .base_controller import BaseController, ControllerConfig

class PDJointPosController(BaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_target(self, action):
        action = action[self.start_index:self.end_index]
        if self.config['use_robot_qlimit']:
            lower, upper = self.qlimits[:, 0], self.qlimits[:, 1]
        else:
            lower, upper = self.config['lower'], self.config['upper'] 
        if self.config['use_delta']:
            if self.config['use_target']:
                self.last_target = target_qpos
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            target_qpos = self._clip_and_scale_action(action, lower, upper)
        
        return target_qpos