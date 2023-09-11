import numpy as np
import math
from envs import TidyUpDish, DrillScrew

SUPPORTED_ARM_TYPE = ['xarm6', 'xarm7']
SUPPORTED_HAND_TYPE = ['allegro']
SUPPORTED_CONTROL_MODE = ['pd_joint_pos', 'pd_ee_pose', 'pd_ee_delta_pose']

def main():
    arm_name = SUPPORTED_ARM_TYPE[1]
    control_mode = SUPPORTED_CONTROL_MODE[1]
    env = TidyUpDish(arm_name=arm_name, control_mode=control_mode)
    env.reset()
    env.viewer.toggle_pause(paused=True) # True
    flag = True

    # 0: left
    # 1: right
    left_initial_pos, left_initial_quat = np.array([-0.422-0.2, -0.445+0.2, 0.094+0.2]), np.array([0.0, 0.0, -0.707, 0.707])
    right_initial_pos, right_initial_quat = np.array([-0.422-0.2, 0.445+0.2, 0.094+0.2]), np.array([0.0, 0.0, 0.707, 0.707]) 
    action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
    action[0, 0:3], action[0, 3:7] = left_initial_pos, left_initial_quat
    action[1, 0:3], action[1, 3:7] = right_initial_pos, right_initial_quat

    while not env.viewer.closed:
        if flag:
            if i < 5: 
                print ("======first pose=====")
                right_pos, right_quat = np.array([-0.622005, -0.445002, 0.293974]), np.array([0.0, 0.0, -0.707, 0.707])
            # # 移动
            elif 5 <= i < 35:   # 35
                print ("======second pose=====")
            #     right_pos, right_quat = np.array([-0.691, 0.189-0.3, 0.421]), np.array([-0.137, -0.238, -0.643, 0.715])
            # elif 10 <= i < 15:
            #     print("======Third Pose=====")
            #     right_pos, right_quat = np.array([-0.794, 0.209-0.3, 0.419]), np.array([-0.137, -0.238, -0.643, 0.715])
            # elif 15 <= i < 20:
            #     print("======Fouth Pose=====")
            #     right_pos, right_quat = np.array([-0.854, 0.154-0.3, 0.311]), np.array([-0.137, -0.238, -0.643, 0.715])
            # elif 20 <= i < 25:
            #     print("======Fifth Pose=====")
            #     right_pos, right_quat = np.array([-0.954, 0.154-0.3, 0.183]), np.array([-0.137, -0.238, -0.643, 0.715])
            # elif 25 <= i < 35:
            #     print("======Sixth Pose=====")
            #     right_pos, right_quat = np.array([-1.000, -0.132, 0.049]), np.array([-0.102, -0.294, -0.542, 0.781])
                right_delta = np.zeros(3)
                arm_ee_key_points = [right_pos, right_quat]
                arm_ee_delta_key_points = [right_pos, right_delta]
                action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
                t = (i-5) / 30
                right_pos = np.array([
                    -0.691 - 0.309*t,
                    0.189 - 0.321*t,
                    0.421 - 0.372*t
                ])
                right_quat = np.array([-0.137, -0.238, -0.643, 0.715])

            # 转手
            elif 35 <= i < 40:
                print("======Seventh Pose=====")
                right_pos, right_quat = np.array([-1.000, -0.132, 0.049]), np.array([-0.197, -0.385, -0.577, 0.693])
            elif 40 <= i < 45:
                print("======Eighth Pose=====")
                right_pos, right_quat = np.array([-1.056, -0.116, -0.020]), np.array([0.197, 0.385, 0.577, -0.692])
            elif 45<= i < 55:
                print("======Nineth Pose=====")
                right_pos, right_quat = np.array([-1.058, -0.090, -0.032]), np.array([-0.395, -0.510, -0.438, 0.626])
                
            # 靠近盘子
            elif 55<= i < 65:
                print("======tenth Pose=====")
                right_pos, right_quat = np.array([-0.989, -0.078, 0.087]), np.array([-0.448, -0.509, -0.540, 0.499])
                right_pos, right_quat = np.array([-1.048, -0.064, -0.040]), np.array([-0.418, -0.536, -0.563, 0.471])
           
            # 转手 -> 用更换control_mode
            elif 65<= i < 95:
                print("======eleventh Pose=====")
                env.switch_control_mode(arm_name=arm_name,control_mode=SUPPORTED_CONTROL_MODE[2])
                action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
                action[0, 2] = 0.2
                
            #     right_pos, right_quat = np.array([-1.049, -0.065, -0.040]), np.array([-0.352, -0.619, -0.474, 0.517])
            # elif 75<= i < 90:
            #     print("======twelveth Pose=====")
            #     right_pos, right_quat = np.array([-1.049, -0.065, -0.040]), np.array([-0.308, -0.663, -0.504, 0.460])
            # elif 90 <= i < 100:
            #     print("======thirtheenth Pose=====")
            #     right_pos, right_quat = np.array([-1.049, -0.065, -0.040]), np.array([-0.155, -0.765, -0.570, 0.257])
                t = (i-65)/30
                right_pos = np.array([-1.049, -0.065, -0.040])                
                right_quat = [
                -0.365 + 0.210*t,
                -0.489 + 0.130*t,
                -0.588 + 0.114*t,
                0.499 - 0.242*t
                ]
            # right_pos = np.zeros(3)
            # right_quat =
            action[0, 0:3], action[0, 3:7] = right_pos, right_quat
            if i==100:
                flag = False
            i+=1

        # traj = Trajectory(config, control_mode)
        # arm_pose = traj.get_arm_key_points()
        # hand_pose = traj.get_hand_key_points()
        # action(t) = arm_pose(t) + hand_pose(t)
        else:
            pass
        env.step(action)
        print("-------Action---------\n", action[0])
        print("-------pose-----------\n", env.get_ee_pose()[0],"\n")
        # print(env.get_ee_pose())


if __name__ == '__main__':
    main()