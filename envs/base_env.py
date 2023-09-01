import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from utils.controller_utils import set_up_controller
from utils.robot_utils import load_robot, generate_robot_info, recover_action, compute_inverse_kinematics, get_kinematic_model



class BaseEnv():
    def __init__(self):
        self._init_engine_renderer()
        self._init_scene()
        self._init_viewer()

    def _init_engine_renderer(self):
        self._engine = sapien.Engine()
        self._renderer = sapien.SapienRenderer()
        self._engine.set_renderer(self._renderer)
        self._engine.set_log_level("error")

    def _init_scene(self):
        self._simulation_freq = 500
        self._scene = self._engine.create_scene()
        self._scene.set_timestep(1 / self._simulation_freq)  # Simulate in 500Hz
        self._add_background()
        self._add_table()
        self._add_agent()
        self._add_workspace()
        self._add_actor()
        
    def _add_background(self):
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material
        render_material = self._renderer.create_material()
        render_material.set_base_color([0.21, 0.18, 0.14, 1.0])
        self._scene.add_ground(altitude = -1.0, render_material = render_material, render_half_size=[8,8])
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    def _add_table(self, pose=sapien.Pose(p=[-0.05, 0, -0.20]), length=0.5, width=1.83, height=0.77, thickness=0.03, color=(0.8, 0.6, 0.4), name='table'):
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

    def _add_agent(self, fix_root_link=True, x_offset=-0.15, y_offset=0.4):
        self._init_control_property()   # initialize control property before adding robots.
        # NOTE(chichu): allegro hands used here have longer customized finger tips
        # TODO(chichu): add xarm7 and ability hand.
        self.robot_left = load_robot(self._scene, 'robot_left')
        self.robot_left.set_root_pose(sapien.Pose([x_offset, -y_offset, -0.20], [0, 0, 0, 1]))
        # self.kinematic_model_left = get_kinematic_model(self.robot_left)
        self.controller_robot_left = set_up_controller(arm_name='xarm6', hand_name='allegro', 
                                                       control_mode=self._control_mode, robot=self.robot_left)
        self.robot_right = load_robot(self._scene, 'robot_right')
        self.robot_right.set_root_pose(sapien.Pose([x_offset, y_offset, -0.20], [0, 0, 0, 1]))
        self.controller_robot_right = set_up_controller(arm_name='xarm6', hand_name='allegro', control_mode=self._control_mode,
                                                        robot=self.robot_right)
        self.robot = [self.robot_left, self.robot_right]
        self.controller = [self.controller_robot_left, self.controller_robot_right]
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
        # TODO(chichu): add joint control, delta pose control based on the current target version
        self._control_mode = control_mode
        self._control_freq = control_freq
        assert (self._simulation_freq % self._control_freq == 0)
        self._frame_skip = self._simulation_freq // self._control_freq
        self._control_time_step = 1 / self._control_freq
        # NOTE(chichu): pid gains are set in load_robot() function.

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
        self._before_control_step()
        self._set_target(action)
        for _ in range(self._frame_skip):
            self._before_simulation_step()
            self._simulation_step()
            self._after_simulation_step()
        self._after_control_step()

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
            # Use inverse kinematics to calculate target arm_qpos
            # action = np.clip(action, -1, 1)
            # self.target_root_velocity[index] = (recover_action(action[:6], self.velocity_limit[index][:6]))
            # palm_jacobian = self.kinematic_model[index].compute_end_link_spatial_jacobian(self.current_qpos[index][:self.arm_dof[index]])
            # arm_qvel = compute_inverse_kinematics(self.target_root_velocity[index], palm_jacobian)[:self.arm_dof[index]]
            # arm_qvel = np.clip(arm_qvel, -np.pi / 1, np.pi / 1)
            # arm_qpos = arm_qvel * self._control_time_step + self.robot[index].get_qpos()[:self.arm_dof[index]]
            # hand_qpos = recover_action(action[6:], self.robot[index].get_qlimits()[self.arm_dof[index]:])
            # target_qpos = np.concatenate([arm_qpos, hand_qpos])
            # target_qvel = np.zeros_like(target_qpos)
            # target_qvel[:self.arm_dof[index]] = arm_qvel
            # self.robot[index].set_drive_target(target_qpos)
            # self.robot[index].set_drive_velocity_target(target_qvel)

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

    def _simulation_step(self):
        self._scene.step()
        self._scene.update_render()
        self.viewer.render()

    def _after_simulation_step(self):
        pass

    def _init_cache_robot_info(self, root_frame='robot'):
        self.robot_name = ['robot_left', 'robot_right']
        self.robot_info = generate_robot_info()
        self.arm_dof, self.hand_dof = [], []
        self.velocity_limit, self.kinematic_model, self.robot_collision_links = [], [], []
        self.root_frame, self.base_frame_pos = [], []
        self.ee_link_name, self.ee_link = [], []
        self.finger_tip_links, self.finger_contact_links = [], []
        self.finger_contact_ids, self.finger_tip_pos = [], []
        self.object_in_tip, self.target_in_object, self.target_in_object_angle = [], [], []
        self.object_lift = []
        self.robot_object_contact = []
        self.ee_link_last_pose, self.ee_link_new_pose = [], []
        self.current_qpos = []
        self.target_root_velocity = []
        self.cartesian_error = []
        for i, name in enumerate(self.robot_name):
            self.arm_dof.append(self.robot_info[name].arm_dof)
            self.hand_dof.append(self.robot_info[name].hand_dof)
            
            velocity_limit = np.array([1] * self.arm_dof[i] + [np.pi] * self.hand_dof[i])
            self.velocity_limit.append(np.stack([-velocity_limit, velocity_limit], axis=1))
            
            # start_joint_name = self.robot[i].get_joints()[1].get_name()
            # end_joint_name = self.robot[i].get_active_joints()[self.arm_dof[i] - 1].get_name()
            # self.kinematic_model.append(PartialKinematicModel(self.robot[i], start_joint_name, end_joint_name))
            
            self.robot_collision_links.append([link for link in self.robot[i].get_links() if len(link.get_collision_shapes()) > 0])
            
            self.root_frame.append(root_frame)
            self.base_frame_pos.append(np.zeros(3))

            # self.ee_link_name.append(self.kinematic_model[i].end_link_name)
            # self.ee_link.append([link for link in self.robot[i].get_links() if link.get_name() == self.ee_link_name[0]][0])
            
            # self.palm_link_name.append(self.robot_info[name].palm_name)
            # self.palm_link.append([link for link in self.robot[i].get_links() if link.get_name() == self.palm_link_name[0]][0])

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
            
            self.object_in_tip.append(np.zeros([len(finger_tip_names), 3]))
            self.target_in_object.append(np.zeros([3]))
            self.target_in_object_angle.append(np.zeros([1]))
            self.object_lift.append(0)

            # Contact buffer
            self.robot_object_contact.append(np.zeros(len(finger_tip_names) + 1))
            
            self.ee_link_last_pose.append(0)
            self.current_qpos.append(0)
            self.target_root_velocity.append(0)
            self.cartesian_error.append(0)