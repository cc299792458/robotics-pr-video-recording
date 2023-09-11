import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from utils.trajectory_utils import Trajectory
from utils.controller_utils import set_up_controller
from utils.robot_utils import load_robot, generate_robot_info


class BaseEnv():
    def __init__(self, arm_name='xarm6', hand_name='allegro', control_mode='pd_joint_pos', time_interval = 10):
        self._init_engine_renderer()
        self._init_scene(arm_name, hand_name, control_mode)
        self._init_trajectory(arm_name, hand_name, time_interval)
        self._init_viewer()

    def _init_engine_renderer(self):
        self._engine = sapien.Engine()
        self._renderer = sapien.SapienRenderer()
        self._engine.set_renderer(self._renderer)
        self._engine.set_log_level("error")

    def _init_scene(self, arm_name, hand_name, control_mode):
        self._simulation_freq = 500
        self._scene = self._engine.create_scene()
        self._scene.set_timestep(1 / self._simulation_freq)  # Simulate in 500Hz
        self._add_background()
        self._add_table()
        self._add_agent(arm_name, hand_name, control_mode)
        self._add_workspace()
        self._add_actor()

    def _init_trajectory(self, arm_name, hand_name, time_interval):
        self.traj = Trajectory(arm_name, hand_name, time_interval, qlimit_scope = self._qlimit_scope)
        
    def _add_background(self):
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material
        render_material = self._renderer.create_material()
        render_material.set_base_color([0.21, 0.18, 0.14, 1.0]) # type: ignore
        self._scene.add_ground(altitude = -1.0, render_material = render_material, render_half_size=[8,8]) # type: ignore
        self._scene.set_ambient_light([0.5, 0.5, 0.5]) # type: ignore
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5]) # type: ignore

    def _add_table(self, pose=sapien.Pose(p=[-0.05-0.2, 0.2, 0]), length=0.5, width=1.83, height=0.97, thickness=0.03, color=(0.8, 0.6, 0.4), name='table'): # type: ignore
        builder = self._scene.create_actor_builder()
        # Tabletop
        tabletop_pose = sapien.Pose([0., 0., -thickness / 2])  # type: ignore # Make the top surface's z equal to 0
        tabletop_half_size = [length / 2, width / 2, thickness / 2]
        builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size) # type: ignore
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, color=color) # type: ignore
        # Table legs (x4)
        for i in [-1, 1]:
            for j in [-1, 1]:
                x = i * (length - thickness) / 2
                y = j * (width - thickness) / 2
                table_leg_pose = sapien.Pose([x, y, -height / 2]) # type: ignore
                table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
                builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size) # type: ignore
                builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, color=color) # type: ignore
        table = builder.build_static(name=name)
        table.set_pose(pose)
        self.table = table

    def _add_agent(self, arm_name, hand_name, control_mode, x_offset=-0.15-0.2, y_offset=0.35, z_offset=0):
        """
            Initialize control property, build robots and set up controllers.
        """
        self._init_control_property(control_mode=control_mode)   # initialize control property before adding robots.
        # NOTE(chichu): allegro hands used here have longer customized finger tips
        # TODO(chichu): add ability hand.
        robot_name = arm_name + '_' + hand_name + '_hand'
        self.robot_name = [robot_name+'_left', robot_name+'_right']
        self.robot_left = load_robot(self._scene, self.robot_name[0])
        self.robot_left.set_root_pose(sapien.Pose([x_offset, -y_offset+0.2, z_offset], [0, 0, 0, 1])) # type: ignore
        self.controller_robot_left = set_up_controller(arm_name=arm_name, hand_name=hand_name, 
                                                       control_mode=self._control_mode, robot=self.robot_left)
        self.robot_right = load_robot(self._scene, self.robot_name[1])
        self.robot_right.set_root_pose(sapien.Pose([x_offset, y_offset+0.2, z_offset], [0, 0, 0, 1])) # type: ignore
        self.controller_robot_right = set_up_controller(arm_name=arm_name, hand_name=hand_name, 
                                                        control_mode=self._control_mode, robot=self.robot_right)
        self.robot = [self.robot_left, self.robot_right]
        self.controller = [self.controller_robot_left, self.controller_robot_right]
        self._qlimit_scope = self.robot[0].get_qlimits()
        self._time_interval = 10
        self._init_cache_robot_info()

    def _add_workspace(self):
        """ Add workspace.
        """
        raise NotImplementedError

    def _add_actor(self):
        """ Add actors
        """
        raise NotImplementedError

    def _init_viewer(self):
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=2.5, y=1.0, z=1.0)
        self.viewer.set_camera_rpy(r=0, p=-0.5, y = -3.54)
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1.2)
        
    def _init_control_property(self, control_freq=20, control_mode='pd_joint_pos'):
        """
            Initialize basic control propert
            NOTE(chichu): pid gains are set in load_robot() function.
        """
        self._control_mode = control_mode
        self._control_freq = control_freq
        assert (self._simulation_freq % self._control_freq == 0)
        self._frame_skip = self._simulation_freq // self._control_freq
        self._control_time_step = 1 / self._control_freq
        
    def reset(self):
        # Set robot initial qpos
        for index in range(len(self.robot)):
            qpos = np.zeros(self.robot[index].dof)
            xarm_qpos = self.robot_info[self.robot_name[index]].arm_init_qpos
            qpos[:self.arm_dof[index]] = xarm_qpos
            self.robot[index].set_qpos(qpos)
            self.robot[index].set_drive_target(qpos)
        # Reset controller
        for index in range(len(self.controller)):
            self.controller[index].reset()

    def step(self, action):
        self.step_action(action)

    def step_action(self, action):
        if action is not None:
            self._before_control_step()
            self._set_target(action)
        for _ in range(self._frame_skip):
            self._before_simulation_step()
            self._simulation_step()
            self._after_simulation_step()
        self._after_control_step()
    
    def get_ee_pose(self):
        for i in range(len(self.robot)):
            self.ee_pose[i] = [link for link in self.robot[i].get_links() if link.get_name() == self.ee_link_name][0].get_pose()

        return self.ee_pose
    
    def switch_control_mode(self, robot:sapien.Articulation, arm_name, control_mode):
        self._control_mode = control_mode
        if robot.get_name().endswith("left"):
            self.controller_robot_left = set_up_controller(arm_name=arm_name,control_mode=self._control_mode, robot=self.robot_left)
        elif robot.get_name().endswith("right"):
            self.controller_robot_right = set_up_controller(arm_name=arm_name,control_mode=self._control_mode, robot=self.robot_right)
        self.controller = [self.controller_robot_left, self.controller_robot_right]

    def _before_control_step(self):
        return
        for index in range(len(self.robot)):
            self.current_qpos[index] = self.robot[index].get_qpos()
            # print(self.current_qpos[0])
            # self.ee_link_last_pose[index] = (self.ee_link[index].get_pose())

    def _set_target(self, action):
        for index in range(len(self.robot)):
            self.controller[index].set_target(action[index])
        return None

    def _after_control_step(self):
        return None
        for index in range(len(self.robot)):
            ee_link_new_pose = self.ee_link[index].get_pose()
            relative_pos = ee_link_new_pose.p - self.ee_link_last_pose[index].p
            self.cartesian_error[index] = np.linalg.norm(relative_pos - self.target_root_velocity[index][:3] * self._control_time_step)

    def _before_simulation_step(self):
        for robot in self.robot:
            passive_qf = robot.compute_passive_force(external=False)
            robot.set_qf(passive_qf)

    def _simulation_step(self, rd_freq = 1):
        for _ in range(rd_freq):
            self._scene.step()
        self._scene.update_render()
        self.viewer.render()

    def _after_simulation_step(self):
        pass

    def _init_cache_robot_info(self, root_frame='robot'):
        self.robot_info = generate_robot_info()
        self.arm_dof, self.hand_dof = [], []
        self.ee_link_name = 'base_link'
        self.ee_pose = []
        for i, name in enumerate(self.robot_name):
            self.arm_dof.append(self.robot_info[name].arm_dof)
            self.hand_dof.append(self.robot_info[name].hand_dof)
            self.ee_pose.append([link for link in self.robot[i].get_links() if link.get_name() == self.ee_link_name][0].get_pose())
    
    def set_time_interval(self, time_interval):
        self._time_interval = time_interval