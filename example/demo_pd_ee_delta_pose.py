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

    # NOTE(HAOYANG):
    # The frame is the base_link in the palm 
    action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
    while not env.viewer.closed:
        if flag:
            # switch the control mode from pd_ee_pose to pd_ee_delta_pose
            env.switch_control_mode(arm_name=arm_name, control_mode=SUPPORTED_CONTROL_MODE[2])
            action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
            action[0, 2] = 0.2
            flag = False
        else:
            action[0, 2] = 0
            pass

        env.step(action)
        print("-------Action---------\n", action[0])
        print("-------pose-----------\n", env.get_ee_pose()[0],"\n")

if __name__ == '__main__':
    main()