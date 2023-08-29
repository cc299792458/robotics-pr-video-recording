import numpy as np

from typing import Union, Sequence
from .base_controller import BaseController, ControllerConfig

class PDJointVelController(BaseController):
    def __init__(self, control_time_step, kinematic_model, **kwargs):
        super().__init__(**kwargs)
        self._control_time_step = control_time_step
        self._kinematic_model = kinematic_model

    def set_target(self, action):
        self.target_root_velocity = self._clip_and_scale_action(action, self.config.lower, self.config.upper)
        palm_jacobian = self._kinematic_model.compute_end_link_spatial_jacobian(self.qpos[self.start_index:self.end_index])
        target_qvel = self.compute_inverse_kinematics(self.target_root_velocity, palm_jacobian)[self.start_index:self.end_index]
        target_qvel = np.clip(target_qvel, -np.pi / 1, np.pi / 1)
        if self.config.use_target == True:
            pass
        else:
            next_target_qpos = target_qvel * self._control_time_step + self.robot.qpos[self.start_index:self.end_index]

        return next_target_qpos
    
    def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
        lmbda = np.eye(6) * (damping ** 2)
        # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
        delta_qpos = palm_jacobian.T @ \
                        np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

        return delta_qpos

# class PDJointVelControllerConfig(ControllerConfig):
#     lower: Union[None, float, Sequence[float]]
#     upper: Union[None, float, Sequence[float]]
#     use_delta: bool = False
#     use_target: bool = False
#     controller_cls = PDJointVelControllera