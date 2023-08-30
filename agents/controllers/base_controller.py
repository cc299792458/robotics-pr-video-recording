import numpy as np
from dataclasses import dataclass

class BaseController:
    def __init__(self, config, robot, start_index=None, end_index=None, **kwargs):
        self.config = config
        self.robot = robot
        self.start_index = start_index
        self.end_index = end_index
        
    def set_target(self, action):
        raise NotImplementedError

    def _clip_action(self, action, low, high):
        """Clip action to [low, high]."""
        action = np.clip(action, low, high)
        return action

    def _clip_and_scale_action(self, action, low, high):
        """Clip action to [-1, 1] and scale according to a range [low, high]."""
        low, high = np.asarray(low), np.asarray(high)
        action = np.clip(action, -1, 1)
        return 0.5 * (high + low) + 0.5 * (high - low) * action
    
    @property
    def qpos(self):
        """Get current joint positions."""
        return self.robot.get_qpos()[self.start_index:self.end_index]

    @property
    def qvel(self):
        """Get current joint velocities."""
        return self.robot.get_qvel()[self.start_index:self.end_index]
    
    @property
    def qlimits(self):
        """Get qlimits."""
        return self.robot.get_qlimits()[self.start_index:self.end_index]

    # -------------------------------------------------------------------------- #
    # Interfaces (implemented in subclasses)
    # -------------------------------------------------------------------------- #
    def _preprocess_action(self, action: np.ndarray):
        # TODO(jigu): support discrete action
        action_dim = self.action_space.shape[0]
        assert action.shape == (action_dim,), (action.shape, action_dim)
        if self._normalize_action:
            action = self._clip_and_scale_action(action)
        return action

    def set_target(self, action: np.ndarray):
        """Set the action to execute.
        The action can be low-level control signals or high-level abstract commands.
        """
        raise NotImplementedError
    
    def get_target_qpos(self):
        """Get target qpos.
        """
        raise NotImplementedError
    
    def get_target_ee_pose(self):
        """Get target end effector's pose
        """
        raise NotImplementedError
    
    def get_ee_pose(self):
        """Get end effector's pose
        """
        raise NotImplementedError

    def get_state(self) -> dict:
        """Get the controller state."""
        return {}

    def set_state(self, state: dict):
        pass
    
# @dataclass
class ControllerConfig:
    controller_cls = BaseController    

class DictController:
    def __init__(self, arm_name='xarm6', hand_name='allegro', config=None, control_mode=None, robot=None, **kwargs):
        config = config[control_mode]
        self.robot = robot
        if arm_name == 'xarm6':
            arm_dof = 6
        else:
            raise NotImplementedError
        arm_controller_cls = config['arm']['controller_cls']
        arm_controller = arm_controller_cls(config=config['arm'], robot=robot, start_index=0, end_index=arm_dof, **kwargs)
        
        if hand_name == 'allegro':
            hand_dof = 16
        elif hand_name == 'ability':
            hand_dof = 10
        else:
            raise NotImplementedError
        hand_controller_cls = config['hand']['controller_cls']
        hand_controller = hand_controller_cls(config=config['hand'], robot=robot, start_index=arm_dof, end_index=arm_dof+hand_dof, **kwargs)

        self.dict_controller = dict(arm=arm_controller, hand=hand_controller)
    
    def set_target(self, action):
        arm_action = self.dict_controller['arm'].set_target(action)
        hand_action = self.dict_controller['hand'].set_target(action)

        action = np.concatenate([arm_action, hand_action])
        self.robot.set_drive_target(action)
        self.robot.set_drive_velocity_target(np.zeros_like(action))