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
        self._simulation_frequency = 500
        self._scene = self._engine.create_scene()
        self._scene.set_timestep(1 / self._simulation_frequency)  # Simulate in 500Hz
        self._add_background()
        self._add_work_place()
        self._add_agent()
        self._add_actor()
        
    def _add_background(self):
        self._scene.add_ground(altitude=-1.0)
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    def _add_work_place(self, pose=sapien.Pose(p=[-0.05, 0, 0.0]), length=0.4, width=1.0, height=1.0, thickness=0.1, color=(0.8, 0.6, 0.4), name='table'):
        builder = self._scene.create_actor_builder()
        # Tabletop
        tabletop_pose = sapien.Pose([0., 0., -thickness / 2])  # Make the top surface's z equal to 0
        tabletop_half_size = [length / 2, width / 2, thickness / 2]
        builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size)
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, color=color)
        # Table legs (x4)
        for i in [-1, 1]:
            for j in [-1, 1]:
                x = i * (length - thickness) / 2
                y = j * (width - thickness) / 2
                table_leg_pose = sapien.Pose([x, y, -height / 2])
                table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
                builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
                builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, color=color)
        table = builder.build_static(name=name)
        table.set_pose(pose)
        self.table = table

    def _add_agent(self, fix_root_link=True, x_offset=0.05, y_offset=0.25):
        # NOTE(chichu): allegro hands used here have longer customized finger tips
        loader_robot_left = self._scene.create_urdf_loader()
        loader_robot_left.fix_root_link = fix_root_link
        self.robot_left = loader_robot_left.load("./assets/robot/xarm6_description/xarm6_allegro_long_finger_tip_left.urdf")
        self.robot_left.set_root_pose(sapien.Pose([x_offset, y_offset, 0.0], [1, 0, 0, 0]))
        loader_robot_right = self._scene.create_urdf_loader()
        loader_robot_right.fix_root_link = fix_root_link
        self.robot_right = loader_robot_right.load("./assets/robot/xarm6_description/xarm6_allegro_long_finger_tip_right.urdf")
        self.robot_right.set_root_pose(sapien.Pose([x_offset, -y_offset, 0.0], [1, 0, 0, 0]))
        
        self.robot = [self.robot_left, self.robot_right]

    def _add_actor(self):
        """ Add actors
        """
        raise NotImplementedError
    
    def _init_viewer(self):
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=-0.8, y=0, z=1.0)
        self.viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    def _init_controller(self, control_frequency=20):
        # TODO(chichu): add controllers here
        self._control_frequency = control_frequency
        assert (self._simulation_frequency % self._control_frequency == 0)
        self._frame_skip = self._simulation_frequency // self._control_frequency
        pass

    def reset(self):
        self._init_scene()

    def step(self, action):
        self.step_action(action)

    def step_action(self, action):
        self._before_control_step()
        for _ in range(self._frame_skip):
            self._before_simulation_step()
            self._simulation_step()
            self._after_simulation_step()
        self._after_control_step()

    def _before_control_step(self):
        pass

    def _after_control_step(self):
        pass

    def _before_simulation_step(self):
        for robot in self.robot:
            passive_qf = robot.compute_passive_force(external=False)
            robot.set_qf(passive_qf)

    def _simulation_step(self):
        self._scene.step()
        self._scene.update_render()
        self.viewer.render()

    def _after_simulation_step(self):
        pass