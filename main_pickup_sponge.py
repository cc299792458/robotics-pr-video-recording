from PIL import Image
import numpy as np
import math
from envs import TidyUpDish, DrillScrew
from utils.recording_gif import record_gif
SUPPORTED_ARM_TYPE = ['xarm6', 'xarm7']
SUPPORTED_HAND_TYPE = ['allegro','ability']
SUPPORTED_CONTROL_MODE = ['pd_joint_pos', 'pd_ee_pose', 'pd_ee_delta_pose']
TIME_INTERVAL = [5,10,15,20]

def normalize_hand_joints(env, hand_joints):
    qlimit_scope = env._qlimit_scope[env.arm_dof[0]:]
    print(qlimit_scope)
    normalized_joints = np.zeros_like(hand_joints)
    for i, (q_min, q_max) in enumerate(qlimit_scope):
        normalized_joints[i] = (2 * (hand_joints[i] - q_min) / (q_max - q_min)) - 1
    return normalized_joints

def main():
    arm_name = SUPPORTED_ARM_TYPE[1]
    hand_name = SUPPORTED_HAND_TYPE[1]
    control_mode = SUPPORTED_CONTROL_MODE[1]
    time_interval = TIME_INTERVAL[1]
    env = TidyUpDish(arm_name=arm_name, hand_name = hand_name, control_mode=control_mode, time_interval = time_interval)
    env.reset()
    env.viewer.toggle_pause(paused=True) # True
    flag = True
    robot_left = env.robot[0]
    robot_right = env.robot[1]

    left_initial_pos = np.array([-0.622, -0.245, 0.294])
    right_initial_pos = np.array([-0.622, 0.645, 0.294])

    left_initial_quat = np.array([0.0, 0.0, -0.707, 0.707])
    right_initial_quat = np.array([0.0, 0.0, 0.707, 0.707])

    # init_arm_action
    if hand_name == 'allegro':
        left_initial_pos, left_initial_quat = np.array([-0.622, -0.245, 0.294]), np.array([0.0, 0.0, -0.707, 0.707])
        right_initial_pos, right_initial_quat = np.array([-0.622, 0.645, 0.294]), np.array([0.0, 0.0, 0.707, 0.707]) 
    elif hand_name == 'ability': 
        left_initial_pos, left_initial_quat = np.array([-0.611, -0.166, 0.294]), np.array([-0.707, 0.0, 0.707, 0])
        right_initial_pos, right_initial_quat = np.array([-0.611, 0.724, 0.294]), np.array([0.0, 0.707, 0.0, -0.707])
    
    # init_hand_action
    if hand_name == 'allegro':
        hand_init_joints = [0.0,-1.0,-1.0,-1.0,0.0,-1.0,-1.0,-1.0,-0.5,-1.0,-1.0,-1.0,1.0,-0.4,-1.0,-0.4]
    elif hand_name == 'ability':
        hand_init_joints = normalize_hand_joints(env=env,hand_joints=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    # init_action
    action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
    action[0, 0:3], action[0, 3:7] = left_initial_pos, left_initial_quat
    action[1, 0:3], action[1, 3:7] = right_initial_pos, right_initial_quat
    
    for i in range(len(env.robot)):
        action[i,env.arm_dof[i]:] = hand_init_joints[:]

    # switch camera
    env.viewer.set_camera_xyz(x=-0.322727, y = 1.32468, z = 0.293172)
    env.viewer.set_camera_rpy(r=0, p=-0.264999983942482, y =2.023186)
    env.viewer.window.set_camera_parameters(near=0.1, far=100, fovy=1.57)
    start_step,end_step = 5, 5 + 4 * time_interval

    while not env.viewer.closed:
        for step in range(end_step+1):
            if flag:
                if start_step <= step < end_step: 
                    key_point_info = env.traj.get_picksponge_stage_key_points(step = step, robot = robot_right)
                    env.switch_control_mode(robot = robot_right, arm_name = arm_name, control_mode=SUPPORTED_CONTROL_MODE[key_point_info["control_mode_index"]]) # type: ignore
                    # action[1, 0:-env.hand_dof[0]], action[1, -env.hand_dof[0]:env.controller[0].action_dim] = key_point_info["arm_pose"], key_point_info["hand_joints"] # type: ignore
                    # env.switch_control_mode(robot = robot_right, arm_name = arm_name, control_mode=SUPPORTED_CONTROL_MODE[0]) # type: ignore
                    # action[1, 0:7] = [-0.148, 0.486, -0.011878177, 0.709, 0.018005097, 0.098, -0.04656761]
                    # save image
                    # rgba = env.viewer.window.get_float_texture('Color')
                    # rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                    # rgba_pil = Image.fromarray(rgba_img)
                    # name = 'ability/images/'+ str(step) + '.png'
                    # rgba_pil.save(name)
                    action[1,-10:] = normalize_hand_joints(env=env,hand_joints=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0])
                elif step == end_step:
                    # record_gif()  # record gif
                    flag = False

            env.step(action)
            print("env.controller[1].action_dim",env.controller[1].action_dim)
            print("step:", step)
            print("-------Action---------\n", action)
            print("-------pose-----------\n", env.get_ee_pose()[1])
            print("-------Joints---------\n", np.array(env.robot_right.get_qpos()[0: env.arm_dof[0]]))

if __name__ == '__main__':
    main()