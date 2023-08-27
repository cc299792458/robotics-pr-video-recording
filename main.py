import numpy as np

from envs import TidyUpDish, DrillScrew


def main():
    env = TidyUpDish()
    env.reset()
    env.viewer.toggle_pause(paused=False) # True
    while not env.viewer.closed:
        action = np.zeros([env.arm_dof[0]+env.hand_dof[1]])
        env.step(action)


if __name__ == '__main__':
    main()