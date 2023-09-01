from agents.controllers import PDJointPosController, PDEEPoseController

def XArmAllegroDefaultConfig(arm_name='xarm6'):
        use_target = True
        arm_dof = int(arm_name[-1])
        hand_dof = 16
        # Arm
        arm_pd_joint_pos = dict(lower=None, upper=None, 
                                normalize_action=False, 
                                use_delta=False,
                                use_target=use_target,
                                action_dim=arm_dof,
                                arm_dof=arm_dof,
                                controller_cls=PDJointPosController)
        arm_pd_ee_pose = dict(lower=None, upper=None, 
                                normalize_action=False,
                                use_delta=False, 
                                use_target=None,
                                action_dim=arm_dof,
                                arm_dof=arm_dof,
                                controller_cls=PDEEPoseController)
        arm_pd_ee_delta_pose = dict(lower=-1, upper=1, rot_bound=1, 
                                    normalize_action=True,
                                    use_delta=True,
                                    use_target=use_target, 
                                    action_dim=6,
                                    arm_dof=arm_dof,
                                    controller_cls=PDEEPoseController)
        # Hand
        hand_pd_joint_pos = dict(lower=None, upper=None, 
                                normalize_action=True, 
                                use_delta=False,
                                use_target=use_target,
                                action_dim=hand_dof,
                                hand_dof=hand_dof,
                                controller_cls=PDJointPosController)

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=hand_pd_joint_pos),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, hand=hand_pd_joint_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, hand=hand_pd_joint_pos),
        )

        return controller_configs