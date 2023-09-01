from agents.controllers import PDJointPosController, PDJointVelController, PDEEPoseController

def XArm6AllegroDefaultConfig():
        use_target = True
        # Arm
        arm_pd_joint_pos = dict(lower=None, upper=None, 
                                normalize_action=False, 
                                use_delta=False,
                                use_target=use_target,
                                start=0, end=6, 
                                controller_cls=PDJointPosController)
        arm_pd_joint_vel = dict(lower=-1, upper=1, 
                                normalize_action=False,
                                use_delta=False, 
                                use_target=use_target,
                                start=0, end=6, 
                                controller_cls=PDJointVelController)
        arm_pd_ee_delta_pose = dict(lower=-0.1, upper=0.1, rot_bound=0.1, 
                                    normalize_action=True,
                                    use_delta=True,
                                    use_target=use_target, 
                                    start=0, end=6,
                                    controller_cls=PDEEPoseController)
        # Hand
        hand_pd_joint_pos = dict(lower=None, upper=None, 
                                normalize_action=True, 
                                use_delta=False,
                                use_target=use_target,
                                start=6, end=22, 
                                controller_cls=PDJointPosController)

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=hand_pd_joint_pos),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, hand=hand_pd_joint_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, hand=hand_pd_joint_pos),
        )

        return controller_configs