from .base_env import BaseEnv

class DrillScrew(BaseEnv):
    def __init__(self):
        super().__init__()

    def _add_background(self):
        super()._add_background()
        # TODO(haoyang): add task related background/scene here.

    def _add_actor(self):
        # TODO(haoyang): add task related actors here.
        pass

    def _add_workspace(self):
        # TODO(haoyang): add task related workspace here, for example, a working table.
        pass