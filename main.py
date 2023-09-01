import numpy as np

from envs import TidyUpDish, DrillScrew

SUPPORTED_ARM_TYPE = ['xarm6', 'xarm7']
SUPPORTED_HAND_TYPE = ['allegro']
SUPPORTED_CONTROL_MODE = ['pd_joint_pos', 'pd_ee_pose', 'pd_ee_delta_pose']

def main():
    arm_name = SUPPORTED_ARM_TYPE[1]
    control_mode = SUPPORTED_CONTROL_MODE[2]
    env = TidyUpDish(arm_name=arm_name, control_mode=control_mode)
    env.reset()
    env.viewer.toggle_pause(paused=True) # True
    flag = True
    while not env.viewer.closed:
        action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
        if flag:
            action[0, 3] = 0.2
            action[1, 3] = 0.2
            flag = False
        else:
            action[0, 3] = 0
            action[1, 3] = 0
        # action[0, 4], action[1, 4] = -np.pi/2, -np.pi/2
        # action[0, 5] = np.pi
        # action = None
        env.step(action)


if __name__ == '__main__':
    main()