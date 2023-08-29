from .base_controller import BaseController
from typing import Union, Sequence


class PDEEPoseController(BaseController):
    def __init__(self):
        pass

    def set_target(self, robot, action):
        pass


# class PDEEPoseControllerConfig(BaseController):
#     lower: Union[None, float, Sequence[float]]
#     upper: Union[None, float, Sequence[float]]
#     use_delta: bool = False
#     use_target: bool = False
#     interpolate: bool = False
#     normalize_action: bool = True
#     controller_cls = PDEEPoseController
