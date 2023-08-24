"""Ant environment.

Notes:
    It is not a full reproduction of Mujoco-based ant.

References:
    https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
"""

import array
import numpy as np
from gym import spaces

import sapien.core as sapien
from sapien.core import Pose
from sapien.utils.viewer import Viewer
from sapien_env import SapienEnv
from robot_pose_control import robot_pose_control
from transforms3d.euler import euler2quat
from transforms3d.euler import quat2euler

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', default='box', choices=['box', 'sphere'], type=str, help='the type of object')

    parser.add_argument('--gravity', default=-9.8, type=float, help='z-axis gravity')
    parser.add_argument('--angle', default=30.0, type=float, help='the angle of the slope')
    parser.add_argument('--offset', default=0.1, type=float, help='z-offset of the slope above the ground')

    parser.add_argument('--static-friction', default=0.3, type=float, help='static friction')
    parser.add_argument('--dynamic-friction', default=0.3, type=float, help='dynamic friction')
    parser.add_argument('--restitution', default=0.1, type=float, help='restitution (elasticity of collision)')
    parser.add_argument('--linear-damping', default=0.0, type=float,
                        help='linear damping (resistance proportional to linear velocity)')
    parser.add_argument('--angular-damping', default=0.0, type=float,
                        help='angular damping (resistance proportional to angular velocity)')
    parser.add_argument('--Tr1', default=-5, type=float,
                        help='the Timelimit1 reward')
    parser.add_argument('--Tr2', default=-100, type=float,
                        help='the Timelimit2 reward')
    args = parser.parse_args()
    return args

# ---------------------------------------------------------------------------- #
# Create the actors
# ---------------------------------------------------------------------------- #
def create_box(
        scene: sapien.Scene,
        pose: sapien.Pose,
        half_size,
        color=None,
        is_kinematic=False,
        density=1000.0,
        physical_material: sapien.PhysicalMaterial = None,
        name='',
) -> sapien.Actor:

    half_size = np.array(half_size)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size, material=physical_material, density=density)  # Add collision shape
    builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
    if is_kinematic:
        box = builder.build_kinematic(name=name)
    else:
        box = builder.build(name=name)
    box.set_pose(pose)
    return box

ee_pose_dis = Pose([0.175, 0.01, 0.280],[1,0,0,0])

args = parse_args()

class GolfEnv(SapienEnv):
    def __init__(self, args):
        super().__init__(control_freq=5, timestep=0.01)

        self.arm_target_init_qpos = [4.71, 3.03, 0.0, 1.02, 4.8, 3.48, 4.88]
        self.gripper_init_qpos = [0.5, 0.8, 0.5, 0.8, 0.5, 0.8]

        self.active_joints = self.robot.get_active_joints()
        self.dof = self.robot.dof
        self._init_state = self.robot.pack()

        # gold_thresh
        self.goal_thresh = 0.085
        self.goal_thresh_1 = 0.15
        self.goal_thresh_2 = 0.2
        self.goal_thresh_3 = 0.3

        # RL space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[2*self.dof + 13], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-3, high=3, shape=[self.dof], dtype=np.float32)
        
        self._per_episo_info=dict(
            is_ball_on_ground = False,
            is_ball_nearby_hole = False,
        )
        # Following the original implementation, we scale the action (qf)
        self._action_scale_factor = 50.0

        self._cur_step = 0
        self._roll_step = 0
        self._max_episode_steps = 500
        self._max_roll_steps = 50
        self.epo_hit_num = 0
        self._epo_hit_step = 0

        self._setup_viewer()
    
    # ---------------------------------------------------------------------------- #
    # Simulation world
    # ---------------------------------------------------------------------------- #
    def _build_world(self):
        # set_scene material
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material

        # set ground (change the material)
        render_material = self._renderer.create_material()
        render_material.set_base_color([0.27, 0.55, 0.0, 1.0])
        self._scene.add_ground(altitude = 0.0, render_material=render_material, render_half_size = [9, 4])

        # set actor
        box_obstacle, blue_flag, golf_ball, golf_stick = self.create_golf_scene(self._scene)
        box_obstacle.set_pose(Pose([2.0, 0.0, 0.0]))
        blue_flag.set_pose(Pose([-4.0, 0.0, 0.05]))
        golf_ball.set_pose(Pose([-4.25, 0.0, 0.4]))


    # @staticmethod
    def create_golf_scene(self, color=(0.8, 0.6, 0.4), 
                   friction=0.0, damping=1.0, density=20.0):
        builder = self._scene.create_actor_builder()
        
        # add a box_obstacle
        half_size = [0.25, 1.0, 0.1]
        box_obstacle = create_box(
            self._scene,
            Pose(p =[0, 0, 0], q =[1, 0, 0, 0],),
            is_kinematic=True,
            half_size = half_size, color=[0., 0.5, 1.], name='box',
        )

        # add a blue_flag
        # builder.add_box_collision(half_size=[0.06, 0.06, 0.05])
        builder.add_visual_from_file(filename='/home/lihaoyang/Project/SuLab/PR-Vedio/assets/kitchen/table/model.dae')
        builder.add_collision_from_file(filename='/home/lihaoyang/Project/SuLab/PR-Vedio/assets/kitchen/table/model.dae')
        blue_flag = builder.build_kinematic(name='blue_flag') # can not be affected by external forces
        
        # add a golf_ball
        builder = self._scene.create_actor_builder()
        builder.add_collision_from_file(scale = [0.02, 0.02, 0.02],density=50, filename='/home/lihaoyang/Project/SuLab/LHY-CN/envs/assets/golf/golf_ball/model.dae')
        builder.add_visual_from_file(scale = [0.02, 0.02, 0.02], filename='/home/lihaoyang/Project/SuLab/SAPIEN/assets/Golf/golf_ball/model.dae')
        golf_ball = builder.build(name='golf_ball')

        # Load URDF
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True     # fix_root_link
        robotbuild: sapien.ArticulationBuilder = loader.load_file_as_articulation_builder("/home/lihaoyang/Project/SuLab/SAPIEN/examples/assets/robot/jaco2/jaco2_new.urdf")
        link_lists = robotbuild.get_link_builders()

        # print the tree 
        print("=============== Without the golf stick ===============")
        for link_idx, link in enumerate(link_lists):
            # print('link_idx',link_idx,': ','link_joint:', link.get_name())
            print(link.get_name())
        print("======================================================",'\n')
        
        golf_stick_robot = robotbuild.build(fix_root_link=True)
        golf_stick_robot.set_name('robot_with_golf_stick')
        golf_stick_robot.set_root_pose(sapien.Pose([-3.525, 2.896, 1.025], [1, 0, 0, 0]))

        return box_obstacle, blue_flag, golf_ball, golf_stick_robot
    
    def step(self, action):
        self._cur_step += 1
        # Use internal velocity drive
        for idx in range(7):
            if idx != 1 or idx != 6:
                self.active_joints[idx].set_drive_velocity_target(action[idx])
        # Control the gripper directly by torque
        qf = self.robot.compute_passive_force(True, True, False)
        # qf[-2:] += action[-2:]
        self.robot.set_qf(qf)
        
        for i in range(self.control_freq):
            self._scene.step()

            # self._scene.update_render()  # 渲染
            # self.render()    # 可视化

        ctr_integral_multi_step = 0
        ball_velocity_value = self.get_velocity_value()

        # Judge by the golf ball's velocity 
        while (ball_velocity_value > 1e-4 or ctr_integral_multi_step % self.control_freq != 0):
            self._scene.step()
            ctr_integral_multi_step +=1 # 保持整数倍
            self.epo_hit_num += 1 # 用来鼓励击打次数
            self._epo_hit_step = self._cur_step # 用来更新timelimit
            ball_velocity_value = self.get_velocity_value()
            if (ctr_integral_multi_step > 1000): 
                break   
            
            # self._scene.update_render()  # 渲染
            # self.render()    # 可视化

        # obtain the related info
        info = self.evaluate()
        obs = self._get_obs()      
        reward = self._get_reward(info)
        done = self._get_done(info)

        return obs, reward, done, info
    
    def reset(self):
        # robot arm
        root_pose = sapien.Pose([-4.4, 0.5, 0], [1, 0, 0, 0])
        arm_target_init_qpos = self.arm_target_init_qpos
        gripper_init_qpos = self.gripper_init_qpos
        robot_pose_control(self.robot,root_pose,arm_target_init_qpos, gripper_init_qpos)        # ——> 需要更改内置的稳定算法，非static target_pose !!!
        
        # ball
        self.ball.set_pose(Pose([-4.25, 0.0, 0.4]))

        # others
        self._cur_step = 0
        self._roll_step = 0
        self._epo_hit_step = 0
        self._per_episo_info=dict(
            is_ball_on_ground = False,
            is_ball_nearby_hole = False,
        )

    def get_velocity_value(self):
        ball_velocity = np.array(self.ball.get_velocity())
        ball_velocity_value = np.linalg.norm(ball_velocity)
        return ball_velocity_value
    
    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):

        self._scene.set_ambient_light([.4, .4, .4])
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])

    def _setup_viewer(self):
        self._setup_lighting()
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=-1.0, y= 4.0, z=2.5)
        self.viewer.set_camera_rpy(y=1.57, p=0.0, r=0)
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        # self.viewer.focus_entity(self.robot)


def pid_forward(pids: list,
                target_pos: np.ndarray, 
                current_pos: np.ndarray, 
                dt: float) -> np.ndarray:
    errors = target_pos - current_pos
    qf = [pid.compute(error, dt) for pid, error in zip(pids, errors)]
    return np.array(qf)

def main():
    args = parse_args()
    env = GolfEnv(args)
    obs = env.reset()
    while not env.viewer.closed:
        for _ in range(1):  # render every 4 steps
            env._scene.step() # 搭建
        env._scene.update_render()  # 渲染
        env.render()    # 可视化

if __name__ == '__main__':
    main()

