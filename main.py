import numpy as np

from envs import TidyUpDish, DrillScrew


def main():
    env = TidyUpDish(arm_name='xarm7', control_mode='pd_ee_delta_pose')
    env.reset()
    env.viewer.toggle_pause(paused=True) # True
    flag = True
    while not env.viewer.closed:
        action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
        if flag:
            action[0, 1] = 0.5
            flag = False
        else:
            action[0, 1] = 0
        # action[0, 4], action[1, 4] = -np.pi/2, -np.pi/2
        # action[0, 5] = np.pi
        # action = None
        env.step(action)


if __name__ == '__main__':
    main()