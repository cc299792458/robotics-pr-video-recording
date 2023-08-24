import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from utils.robot_utils import load_robot, generate_robot_info, recover_action, compute_inverse_kinematics
from kinematics.kinematics_helper import PartialKinematicModel


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
        self._engine.set_log_level("error")

    def _init_scene(self):
        self._simulation_frequency = 500
        self._scene = self._engine.create_scene()
        self._scene.set_timestep(1 / self._simulation_frequency)  # Simulate in 500Hz
        self._add_background()
        self._add_table()
        self._add_agent()
        self._add_actor()
        
    def _add_background(self):
        self._scene.add_ground(altitude=-1.0)
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    def _add_table(self, pose=sapien.Pose(p=[-0.05, 0, 0.0]), length=0.4, width=1.0, height=1.0, thickness=0.1, color=(0.8, 0.6, 0.4), name='table'):
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

    def _add_agent(self, fix_root_link=True, x_offset=0.05, y_offset=0.4):
        # NOTE(chichu): allegro hands used here have longer customized finger tips
        self.robot_left = load_robot(self._scene, 'robot_left')
        self.robot_left.set_root_pose(sapien.Pose([x_offset, y_offset, 0.0], [1, 0, 0, 0]))
        self.robot_right = load_robot(self._scene, 'robot_right')
        self.robot_right.set_root_pose(sapien.Pose([x_offset, -y_offset, 0.0], [1, 0, 0, 0]))
        
        self.robot = [self.robot_left, self.robot_right]
        self._cache_robot_info()

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
        self._control_time_step = 1 / self._control_frequency

    def reset(self):
        self._init_scene()

    def step(self, action):
        self.step_action(action)

    def step_action(self, action):
        self._before_control_step()
        for index in range(len(self.robot)):
            current_qpos = self.robot[index].get_qpos()
            ee_link_last_pose = self.ee_link[index].get_pose()
            action = np.clip(action, -1, 1)
            target_root_velocity = recover_action(action[:6], self.velocity_limit[index][:6])
            palm_jacobian = self.kinematic_model[index].compute_end_link_spatial_jacobian(current_qpos[:self.arm_dof])
            arm_qvel = compute_inverse_kinematics(target_root_velocity, palm_jacobian)[:self.arm_dof[index]]
            arm_qvel = np.clip(arm_qvel, -np.pi / 1, np.pi / 1)
            arm_qpos = arm_qvel * self._control_time_step + self.robot[index].get_qpos()[:self.arm_dof[index]]

            hand_qpos = recover_action(action[6:], self.robot[index].get_qlimits()[self.arm_dof[index]:])
            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            target_qvel = np.zeros_like(target_qpos)
            target_qvel[:self.arm_dof] = arm_qvel
            self.robot[index].set_drive_target(target_qpos)
            self.robot[index].set_drive_velocity_target(target_qvel)
        for _ in range(self._frame_skip):
            self._before_simulation_step()
            self._simulation_step()
            self._after_simulation_step()
        self._after_control_step()
        for index in range(len(self.robot)):
            ee_link_new_pose = self.ee_link[index].get_pose()
            relative_pos = ee_link_new_pose.p - ee_link_last_pose.p
            self.cartesian_error[index] = np.linalg.norm(relative_pos - target_root_velocity[:3] * self.control_time_step)

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

    def _cache_robot_info(self, root_frame='robot'):
        self.robot_name = ['robot_left', 'robot_right']
        self.robot_info = generate_robot_info()
        self.arm_dof, self.hand_dof = [], []
        self.velocity_limit, self.kinematic_model, self.robot_collision_links = [], [], []
        self.root_frame, self.base_frame_pos = [], []
        self.ee_link_name, self.ee_link = [], []
        self.palm_link_name, self.palm_link = [], []
        self.finger_tip_links, self.finger_contact_links = [], []
        self.finger_contact_ids, self.finger_tip_pos = [], []
        self.palm_pose, self.palm_pos_in_base = [], []
        self.object_in_tip, self.target_in_object, self.target_in_object_angle = [], [], []
        self.object_lift = []
        self.robot_object_contact = []
        self.cartesian_error = []
        for i, name in enumerate(self.robot_name):
            self.arm_dof.append(self.robot_info[name].arm_dof)
            self.hand_dof.append(self.robot_info[name].hand_dof)
            
            velocity_limit = np.array([1] * self.arm_dof[i] + [np.pi] * self.hand_dof[i])
            self.velocity_limit.append(np.stack([-velocity_limit, velocity_limit], axis=1))
            
            start_joint_name = self.robot[i].get_joints()[1].get_name()
            end_joint_name = self.robot[i].get_active_joints()[self.arm_dof[i] - 1].get_name()
            self.kinematic_model.append(PartialKinematicModel(self.robot[i], start_joint_name, end_joint_name))
            
            self.robot_collision_links.append([link for link in self.robot[i].get_links() if len(link.get_collision_shapes()) > 0])
            
            self.root_frame.append(root_frame)
            self.base_frame_pos.append(np.zeros(3))

            self.ee_link_name.append(self.kinematic_model[i].end_link_name)
            self.ee_link.append([link for link in self.robot[i].get_links() if link.get_name() == self.ee_link_name][0])
            
            self.palm_link_name.append(self.robot_info[name].palm_name)
            self.palm_link.append([link for link in self.robot[i].get_links() if link.get_name() == self.palm_link_name][0])

            finger_tip_names = (["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"])
            finger_contact_link_name = [
                "link_15.0_tip", "link_15.0", "link_14.0",
                "link_3.0_tip", "link_3.0", "link_2.0", "link_1.0",
                "link_7.0_tip", "link_7.0", "link_6.0", "link_5.0",
                "link_11.0_tip", "link_11.0", "link_10.0", "link_9.0"
            ]
            robot_link_names = [link.get_name() for link in self.robot[i].get_links()]
            self.finger_tip_links.append([self.robot[i].get_links()[robot_link_names.index(name)] for name in finger_tip_names])
            self.finger_contact_links.append([self.robot[i].get_links()[robot_link_names.index(name)] for name in
                                        finger_contact_link_name])
            self.finger_contact_ids.append(np.array([0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4]))
            self.finger_tip_pos.append(np.zeros([len(finger_tip_names), 3]))
            
            self.palm_pose.append(self.palm_link[i].get_pose())
            self.palm_pos_in_base.append(np.zeros(3))
            self.object_in_tip.append(np.zeros([len(finger_tip_names), 3]))
            self.target_in_object.append(np.zeros([3]))
            self.target_in_object_angle.append(np.zeros([1]))
            self.object_lift.append(0)

            # Contact buffer
            self.robot_object_contact.append(np.zeros(len(finger_tip_names) + 1))

            self.cartesian_error.append(0)