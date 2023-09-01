import numpy as np
import sapien.core as sapien

from .base_controller import BaseController
from utils.sapien_utils import get_entity_by_name
from scipy.spatial.transform import Rotation

class PDEEPoseController(BaseController):
    def __init__(self, ee_link_name='base_link', **kwargs):
        """
            Control the end effect's pose.
        """
        super().__init__(**kwargs)
        self._init_pmodel()
        self.ee_link = get_entity_by_name(self.robot.get_links(), ee_link_name)
        self.ee_link_idx = self.robot.get_links().index(self.ee_link)

    def _init_pmodel(self):
        self.pmodel = self.robot.create_pinocchio_model()
        self.qmask = np.zeros(len(self.robot.get_active_joints()), dtype=bool)
        self.qjoint = [joint.get_name() for joint in self.robot.get_active_joints() if 'joint' in joint.get_name() \
                       and 'joint_' not in joint.get_name() and '_joint' not in joint.get_name()]
        all_joint_name = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.qindex = [all_joint_name.index(x) for x in self.qjoint]
        self.qmask[self.qindex] = 1

    def reset(self):
        self._target_qpos = self.qpos
        self._target_pose = self.ee_pose_at_base

    def set_target(self, action):
        """
            Args:
                action: 6 digits in total. 
                        First 3 digits: [delta_x, delta_y, delta_z],
                        Last 3 digits related to rotation.
            Some Options:
                normalize_action: scale the input action to [-1, 1].
                use_delta: calculate next target based on current qpos or last target.
                use_target: calculate next target based on last target. 
        """
        self._target_pose = self.compute_target_pose(action)
        self._target_qpos = self.compute_IK()

        return self._target_qpos
    
    @property
    def ee_pos(self):
        return self.ee_link.pose.p

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.robot.pose.inv()
        return to_base.transform(self.ee_pose)
    
    def _clip_and_scale_action(self, action):
        """
            Clip the pose and rot respectively.
        """
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = super()._clip_and_scale_action(
            action[:3], self.config['lower'], self.config['upper'])
        rot_action = action[3:]
        rot_norm = np.linalg.norm(rot_action)
        if rot_norm != 0:
            rot_action = rot_action / rot_norm
        rot_action = rot_action * self.config['rot_bound']
        return np.hstack([pos_action, rot_action])

    def compute_target_pose(self, action, frame='ee_align'):
        """
            Compute next target pose.
        """
        if self.config['normalize_action']:
            action = self._clip_and_scale_action(action)
        if self.config['use_delta']:
            delta_pos, delta_rot = action[0:3], action[3:6]
            delta_quat = Rotation.from_rotvec(delta_rot).as_quat()[[3, 0, 1, 2]]
            delta_pose = sapien.Pose(delta_pos, delta_quat)
            if self.config['use_target']:
                prev_pose = self._target_pose
            else:
                prev_pose = self.ee_pose_at_base
            if frame == 'base':
                target_pose = delta_pose * prev_pose
            elif frame == 'ee':
                target_pose = prev_pose * delta_pose
            elif frame == "ee_align":
                # origin at ee but base rotation
                target_pose = delta_pose * prev_pose
                target_pose.set_p(prev_pose.p + delta_pos)
        else:
            target_pos, target_rot = action[0:3], action[3:6]
            target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose = sapien.Pose(target_pos, target_quat)

        return target_pose

    def compute_IK(self, max_iterations=1000):
        """
            Compute next target qpos.
        """
        # Assume the target pose is defined in the base frame
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx,
            self._target_pose,
            initial_qpos=self.robot.get_qpos(),
            active_qmask=self.qmask,
            max_iterations=max_iterations,
        )
        if success:
            return result[self.qindex]
        else:
            return self.qpos