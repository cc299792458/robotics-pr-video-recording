from agents.controllers.base_controller import BaseController

class PDJointPosController(BaseController):
    """
        Directly control the qpos of robot.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reset(self):
        self._target_qpos = self.qpos

    def set_target(self, action):
        """
            Args:
                action: action.shape equal to (robot.dof, ).
            Some Options:
                normalize_action: scale the input action to [-1, 1].
                use_delta: calculate next target based on current qpos or last target.
                use_target: calculate next target based on last target. 
        """
        qlimits = self.qlimits
        lower, upper = self.config['lower'], self.config['upper']
        if self.config['normalize_action']:
            if self.config['use_delta']:
                delta_qpos = self._clip_and_scale_action(action, lower, upper)
                if self.config['use_target']:
                    target_qpos = self._target_qpos + delta_qpos
                else:
                    target_qpos = self.qpos + delta_qpos
            else:
                target_qpos = self._clip_and_scale_action(action, qlimits[:, 0], qlimits[:, 1])
        else:
            if self.config['use_delta']:
                delta_qpos = self._clip_action(action, lower, upper)
                if self.config['use_target'] and self._target_qpos != None:
                    target_qpos = self._target_qpos + delta_qpos
                else:
                    target_qpos = self.qpos + delta_qpos
            else:
                target_qpos = self._clip_action(action, qlimits[:, 0], qlimits[:, 1])
        
        self._target_qpos = target_qpos
        return target_qpos