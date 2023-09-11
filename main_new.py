import numpy as np
import math
from envs import TidyUpDish, DrillScrew

SUPPORTED_ARM_TYPE = ['xarm6', 'xarm7']
SUPPORTED_HAND_TYPE = ['allegro']
SUPPORTED_CONTROL_MODE = ['pd_joint_pos', 'pd_ee_pose', 'pd_ee_delta_pose']
TIME_INTERVAL = [5,10,15,20]

def main():
    arm_name = SUPPORTED_ARM_TYPE[1]
    control_mode = SUPPORTED_CONTROL_MODE[1]
    time_interval = TIME_INTERVAL[1]
    env = TidyUpDish(arm_name=arm_name, control_mode=control_mode, time_interval = time_interval)
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
    stat_step,end_step = 5, 5 + 1 * time_interval
    while not env.viewer.closed:
        for step in range(end_step+1):
            if flag:
                if stat_step <= step < end_step: 
                    key_point_info = env.traj.set_up_key_point_property(step = step)
                    env.switch_control_mode(arm_name, control_mode=SUPPORTED_CONTROL_MODE[key_point_info["control_mode_index"]]) # type: ignore
                    action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
                    action[0, 0:-16], action[0, -16:env.controller[0].action_dim] = key_point_info["arm_pose"], key_point_info["hand_joints"] # type: ignore
                elif step == end_step:
                    flag = False
            else:
                # break
                pass
            env.step(action)
            print("env.controller[0].action_dim",env.controller[0].action_dim)
            print("step:", step)
            print("-------Action---------\n", action)
            print("-------pose-----------\n", env.get_ee_pose()[0],"\n")

if __name__ == '__main__':
    main()