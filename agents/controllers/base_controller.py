import numpy as np

class BaseController:
    def __init__(self, config, robot, start, end, **kwargs):
        self.config = config
        self.robot = robot
        self.start_index = start
        self.end_index = end
        
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
    
    def reset(self):
        """Reset the controller.
        """
        raise NotImplementedError

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

class DictController:
    def __init__(self, config=None, control_mode=None, robot=None, **kwargs):
        config = config[control_mode]
        self.robot = robot

        arm_dof, hand_dof = config['arm']['arm_dof'], config['hand']['hand_dof']
        self.robot_dof = arm_dof + hand_dof

        arm_action_dim, hand_action_dim = config['arm']['action_dim'], config['hand']['action_dim']
        self.action_dim = arm_action_dim + hand_action_dim
        self.action_mapping = dict(arm=[0, arm_action_dim], hand=[arm_action_dim, self.action_dim])

        arm_controller_cls = config['arm']['controller_cls']
        arm_controller = arm_controller_cls(config=config['arm'], robot=robot, 
                                            start=0, end=arm_dof, **kwargs)
        hand_controller_cls = config['hand']['controller_cls']
        hand_controller = hand_controller_cls(config=config['hand'], robot=robot, 
                                            start=arm_dof, end=self.robot_dof, **kwargs)

        self.dict_controller = dict(arm=arm_controller, hand=hand_controller)
    
    def set_target(self, action):
        arm_action = self.dict_controller['arm'].set_target(action[self.action_mapping['arm'][0]:self.action_mapping['arm'][1]])
        hand_action = self.dict_controller['hand'].set_target(action[self.action_mapping['hand'][0]:self.action_mapping['hand'][1]])

        action = np.concatenate([arm_action, hand_action])
        self.robot.set_drive_target(action)
        self.robot.set_drive_velocity_target(np.zeros_like(action))

    def reset(self):
        self.dict_controller['arm'].reset()
        self.dict_controller['hand'].reset()