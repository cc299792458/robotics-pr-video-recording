import numpy as np

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
    left_initial_pos, left_initial_quat = np.array([-0.422, -0.445, 0.094]), np.array([0.0, 0.0, -0.707, 0.707])
    right_initial_pos, right_initial_quat = np.array([-0.422, 0.445, 0.094]), np.array([0.0, 0.0, 0.707, 0.707]) 
    action = np.zeros([(env.controller[0].action_dim)])[np.newaxis, :].repeat(2, axis=0)
    action[0, 0:3], action[0, 3:7] = left_initial_pos, left_initial_quat
    action[1, 0:3], action[1, 3:7] = right_initial_pos, right_initial_quat
    while not env.viewer.closed:
        if flag:
            action[0, 2] += 0.2
            action[1, 2] += 0.2
            flag = False
        else:
            pass
            # action[0, 3] = 0
            # action[1, 3] = 0
        # action[0, 4], action[1, 4] = -np.pi/2, -np.pi/2
        # action[0, 5] = np.pi
        # action = None
        env.step(action)
        print(env.get_ee_pose())


if __name__ == '__main__':
    main()