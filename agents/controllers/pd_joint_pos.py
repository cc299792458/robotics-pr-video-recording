import numpy as np

from typing import Union, Sequence
from agents.controllers.base_controller import BaseController, ControllerConfig

class PDJointPosController(BaseController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_qpos = None

    def set_target(self, action):
        action = action[self.start_index:self.end_index]
        qlimits = self.qlimits
        lower, upper = self.config['lower'], self.config['upper']
        if self.config['normalize_action']:
            if self.config['use_delta']:
                delta_qpos = self._clip_and_scale_action(action, lower, upper)
                if self.config['use_target'] and self.target_qpos != None:
                    target_qpos = self.target_qpos + delta_qpos
                else:
                    target_qpos = self.qpos + delta_qpos
            else:
                target_qpos = self._clip_and_scale_action(action, qlimits[:, 0], qlimits[:, 1])
        else:
            if self.config['use_delta']:
                delta_qpos = self._clip_action(action, lower, upper)
                if self.config['use_target'] and self.target_qpos != None:
                    target_qpos = self.target_qpos + delta_qpos
                else:
                    target_qpos = self.qpos + delta_qpos
            else:
                target_qpos = self._clip_action(action, qlimits[:, 0], qlimits[:, 1])
        
        self.target_qpos = target_qpos
        return target_qpos