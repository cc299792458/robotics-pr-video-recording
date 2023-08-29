from agents.controllers import PDJointPosController, PDJointVelController, PDEEPoseController

def XArmAllegroDefaultConfig():
        use_target = True
        # Arm
        arm_pd_joint_pos = dict(lower=-1, upper=1, 
                                use_robot_qlimit=False, 
                                use_delta=False,
                                use_target=use_target, 
                                controller_cls=PDJointPosController)
        arm_pd_joint_vel = dict(lower=-1, upper=1, 
                                use_robot_qlimit=False,
                                use_delta=False, 
                                use_target=use_target, 
                                controller_cls=PDJointVelController)
        arm_pd_ee_delta_pose = dict(lower=-0.1, upper=0.1, 
                                    use_robot_qlimit=False,
                                    use_delta=True,
                                    use_target=use_target, 
                                    controller_cls=PDEEPoseController)
        # Hand
        finger_pd_joint_pos = dict(lower=None, upper=None, 
                                   use_robot_qlimit=True, 
                                   use_delta=False,
                                   use_target=use_target, 
                                   controller_cls=PDJointPosController)

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=finger_pd_joint_pos),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, hand=finger_pd_joint_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, hand=finger_pd_joint_pos),
        )

        return controller_configs