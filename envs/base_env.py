import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer


class BaseEnv():
    def __init__(self):

        self._init_engine_renderer()
        self._init_scene()
        self._init_viewer()
        self._init_controller()

    def _init_engine_renderer(self):
        self._engine = sapien.Engine()
        self._renderer = sapien.SapienRenderer()
        self._engine.set_renderer(self._renderer)

    def _init_scene(self):
        self._scene = self._engine.create_scene()
        self._scene.set_timestep(1 / 500.0)  # Simulate in 500Hz
        self._add_background()
        self._add_agent()
        self._add_actor()
        
    def _add_background(self):
        self._scene.add_ground(altitude=-1.0)
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    def _add_agent(self):
        # TODO(chichu): add agents here
        pass

    def _add_actor(self):
        """ Add actors
        """
        raise NotImplementedError

    def _init_viewer(self):
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=-0.8, y=0, z=0.6)
        self.viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    def _init_controller(self):
        # TODO(chichu): add controllers here
        pass

    def reset(self):
        self._init_scene()

    def step(self, action):
        self.step_action(action)

    def step_action(self, action):
        self._scene.step()
        self._scene.update_render()
        self.viewer.render()
