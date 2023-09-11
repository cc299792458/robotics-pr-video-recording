import numpy as np
from sapien.core import Pose
from numpy import pi
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien

class Trajectory():
  def __init__(self,arm_name='xarm7', hand_name = 'ability', time_interval = 10, qlimit_scope = None):
    self.arm_dof = int(arm_name[-1])
    self.hand_name = hand_name
    self.arm_name = arm_name
    self._hand_init_joints = None

    self.time_interval = time_interval
    if hand_name == 'ability':
      self.hand_dof = 10
    elif hand_name == 'allego': 
      self.hand_dof = 16

    self._qlimit_scope = qlimit_scope

  def get_pickup_stage_key_points(self, step):
    # """get close to the plate"""
    if 5 <= step <= 5 + self.time_interval:
      print("=== Phase: 1 ===")
      step = (step-5) / self.time_interval
      control_mode_index = 1 
      arm_trans = self.interpolate_pos(step, np.array([-0.691, -0.16, 0.385]), np.array([-1.000, -0.132, 0.149]), 0, 1)
      arm_rot = self.slerp_quat(step, np.array([-0.137, -0.238, -0.643, 0.715]), np.array([-0.3011, -0.384, -0.6089, 0.6249]), 0, 1)
      hand_joints = self._hand_init_joints
      arm_joints = np.zeros(self.arm_dof)
    elif 5 + self.time_interval < step <= 5 + 2 * self.time_interval:
      print("=== Phase: 2 ===")
      step = (step-(5 + self.time_interval)) / self.time_interval
      control_mode_index = 1
      arm_trans = self.interpolate_pos(step, np.array([-1.000, -0.132, 0.149]), np.array([-1.048, -0.121, -0.056]), 0, 1)
      arm_rot = self.slerp_quat(step, np.array([-0.3011, -0.384, -0.6089, 0.6249]), np.array([-0.418, -0.536, -0.563, 0.471]), 0, 1)
    
      hand_prep_joints = self.normalize_hand_joints(np.array([-0.014113475, 0.987, 0.795, 
                                         0.917, -0.003, 0.913, 0.77353936, 
                                         0.917, -0.0057257833, 0.976, 0.7593499, 
                                         0.917, 0.786, 0.444, -0.028, 0.762]))
      
      hand_joints = self.interpolate_pos(step, self._hand_init_joints, hand_prep_joints, 0, 1)
      
    # hand_init
    elif 5 + 2 * self.time_interval < step <= 5 + 3 * self.time_interval:
      print("=== Phase: 3 ===")
      step = (step-(5 + 2 * self.time_interval))/ self.time_interval
      control_mode_index = 1
      arm_trans = self.interpolate_pos(step, np.array([-1.048, -0.121, -0.056]), np.array([-1.052, -0.051, -0.019]), 0, 1) # -1.048, -0.101, -0.056
      arm_rot = self.slerp_quat(step,np.array([-0.418, -0.536, -0.563, 0.471]), np.array([-0.439, -0.506, -0.544, 0.505]), 0, 1)  # -0.484, -0.500, -0.500, 0.515                 
      hand_joints = self._hand_init_joints



    elif 5 + 3 * self.time_interval <= step < 5 + 4 * self.time_interval:
      print("=== Phase: 4 ===")
      step = (step-(5 + 3 * self.time_interval))/ self.time_interval
      control_mode_index = 1
      arm_trans = self.interpolate_pos(step, np.array([-1.048, -0.101, -0.056]), np.array([-1.049, -0.034, -0.048]), 0, 1)
      arm_rot = self.slerp_quat(step, np.array([-0.484, -0.500, -0.500, 0.515]), np.array([-0.482, -0.482, -0.503, 0.531]),0, 1)

      hand_prev_joints = self.normalize_hand_joints(np.array([-0.014113475, 0.987, 0.795, 
                                         0.917, -0.003, 0.913, 0.77353936, 
                                         0.917, -0.0057257833, 0.976, 0.7593499, 
                                         0.917, 0.786, 0.444, -0.028, 0.762]))
      
      hand_next_joints = self.normalize_hand_joints(np.array([0.0004900102, 0.7078395, 1.161, 
                                                              1.055, -0.0017214618, 0.70846146, 1.1254236, 
                                                              1.068, -0.0017494204, 0.70876163, 1.121, 
                                                              1.095, 0.801, 0.4170815, 0.266, 0.985]))
      hand_joints = self.interpolate_pos(step, hand_prev_joints, hand_next_joints, 0, 1)

    # hand_catch
    elif 40 <= step < 50:
      print("=== Phase: 5 ===")
      step = (step-40)/10
      control_mode_index = 1
      arm_trans = self.interpolate_pos(step, np.array([-1.047, -0.066, -0.046]), np.array([-1.047, -0.047, -0.064]), 0, 1)
      arm_rot = self.slerp_quat(step,np.array([-0.418, -0.536, -0.563, 0.471]), np.array([-0.294, -0.618, -0.636, 0.356]), 0, 1)
      hand_init_joints = self.normalize_hand_joints(np.array([0, 0.45, 0.491, 
                                  0.6955037, 4.3072873e-08, 0.442, 0.483, 
                                  0.6954995, 9.509424e-08, 0.473, 0.451, 
                                  0.69549584, 1.121, 0.041, -0.134, 0.7785038]))
      
      hand_end_joints = self.normalize_hand_joints(np.array([0, 0.811, 0.491, 
                                  1.296, 4.3072873e-08, 0.803, 0.483, 
                                  1.273, 9.509424e-08, 0.903, 0.451, 
                                  1.312, 1.073, 0.451, -0.134, 0.7785038]))
      
      hand_joints = self.interpolate_pos(step, hand_init_joints, hand_end_joints, 0, 1) 

    else:
      print("Wrong step")

    if control_mode_index == 1:
      arm_pose = np.zeros(7)
      arm_pose[0:3], arm_pose[3:7] = arm_trans, arm_rot
    elif control_mode_index == 2:
      arm_pose = np.zeros(6)
      arm_pose[0:3], arm_pose[3:6] = arm_trans, arm_rot
    elif control_mode_index == 0:
      arm_pose = np.zeros(self.arm_dof)
      arm_pose[0:self.arm_dof] = arm_joints
      
    else:
      raise NotImplementedError
    
    key_point_info = dict(
        arm_pose=arm_pose,
        hand_joints= hand_joints,
        control_mode_index= control_mode_index
      )
    return key_point_info

  def get_picksponge_stage_key_points(self, step, robot:sapien.Articulation):
    # obtain_actual_start_pose
    if (step-5) % self.time_interval == 0 :
      self._cur_arm_qpos = self.normalize_arm_joints(np.array(robot.get_qpos()[0: self.arm_dof]))
      self._cur_hand_qpos = self.normalize_arm_joints(np.array(robot.get_qpos()[self.arm_dof:]))
      
    if self.hand_name == 'allegro':
      self._hand_init_joints = self.normalize_hand_joints(np.array([-0.343, 0.75430614, 0.7220109, 0.69363683, -0.169, 0.74, 0.733, 0.6402707, -0.040042527, 0.613, 0.6241672, 0.90162325, 1.3960001, 0.087, -0.18107991, -0.01]))
      if 5 <= step <= 5 + self.time_interval:
        print("=== Phase: 1 ===")
        control_mode_index = 0
        step = (step-5) / self.time_interval
        arm_next_joints = self.normalize_arm_joints(np.array([-0.19848008, 0.206, -0.050032955, 1.084, 0.004846578, 1.156, -1.925]))
        arm_joints = arm_next_joints  # you can choose whether to use interpolation
        hand_joints = self._hand_init_joints  # you can choose whether to use interpolation

      elif 5 + self.time_interval < step <= 5 + 2 * self.time_interval:
        step = (step-(5 + self.time_interval)) / self.time_interval
        print("=== Phase: 2 ===")
        control_mode_index = 0
        arm_next_joints = self.normalize_arm_joints(np.array([-0.19572468, 0.388, -0.05415525, 1.0347804, -0.009813051, 0.895, -1.8049147]))
        arm_joints = arm_next_joints # you can choose whether to use interpolation
        hand_next_joints = self.normalize_hand_joints(np.array([-0.09608228, 0.777, 0.64630485, 
                                                                0.72204995, -0.044807687, 0.674, 0.8609028, 
                                                                0.6446226, 0.034087706, 0.747, 0.5746397, 
                                                                0.9278148, 1.3719177, 0.28595054, 0.0, 0.30101454]))
        hand_joints = hand_next_joints # you can choose whether to use interpolation

      elif 5 + 2 * self.time_interval < step <= 5 + 3 * self.time_interval:
        print("=== Phase: 3 ===")
        step = (step-(5 + 2 * self.time_interval)) / self.time_interval
        control_mode_index = 1
        arm_trans = np.array([-0.840,0.669,0.076])
        arm_rot = np.array([-0.019,-0.827,0.040,0.561])
        hand_prev_joints = self.normalize_hand_joints(np.array([-0.09608228, 0.777, 0.64630485, 
                                                                0.72204995, -0.044807687, 0.674, 0.8609028, 
                                                                0.6446226, 0.034087706, 0.747, 0.5746397, 
                                                                0.9278148, 1.3719177, 0.28595054, 0.0, 0.30101454]))
        hand_next_joints = self.normalize_hand_joints(np.array([-0.027, 0.862, 0.625, 
                                                                0.73, -0.010002386, 0.801, 0.8966845, 
                                                                0.64584666, 0.055, 0.947, 0.561, 
                                                                0.935, 1.3959993, 0.335, -0.084, 0.877]))
        hand_joints = self.interpolate_pos(step, hand_prev_joints, hand_next_joints, 0, 1)

      elif 5 + 3 * self.time_interval < step <= 5 + 4 * self.time_interval:
        print("=== Phase: 4 ===")
        target_velocity = np.zeros(self.arm_dof+self.hand_dof)
        target_velocity[self.arm_dof:] = 1

        robot.set_drive_velocity_target(target_velocity)
        step = (step-(5 + 3 * self.time_interval)) / self.time_interval
        control_mode_index = 1
        arm_trans = np.array([-0.840,0.669,0.282])
        arm_rot = np.array([-0.019,-0.827,0.040,0.561])

        hand_next_joints = self.normalize_hand_joints(np.array([-0.027, 0.862, 0.625, 
                                                                0.73, -0.010002386, 0.801, 0.8966845, 
                                                                0.64584666, 0.055, 0.947, 0.561, 
                                                                0.935, 1.3959993, 0.335, -0.084, 0.877]))
        hand_joints = hand_next_joints

    elif self.hand_name == 'ability':
      self._hand_init_joints = self.normalize_hand_joints([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]) 
      if 5 <= step <= 5 + self.time_interval:
            print("=== Phase: 1 ===")
            control_mode_index = 0
            step = (step-5) / self.time_interval
            arm_trans = self.interpolate_pos(step, np.array([-0.611, 0.724, 0.294]), np.array([-0.689, 0.650, 0.161]), 0, 1)
            arm_rot = self.slerp_quat(step, np.array([0.0, 0.707, 0.0, -0.707]), np.array([-0.538, 0.461, 0.525, -0.472]), 0, 1)
            arm_next_joints = self.normalize_arm_joints(np.array([0.063, 1.983, -1.671, 2.166, 0.402, -0.96, -0.063]))
            # arm_next_joints = self.interpolate_pos(step, self._cur_arm_qpos, arm_next_joints, 0, 1)
            arm_joints = arm_next_joints
            hand_joints = self._hand_init_joints

      elif 5 + self.time_interval < step <= 5 + 2 * self.time_interval:
        print("=== Phase: 2 ===")
        step = (step-(5 + self.time_interval)) / self.time_interval
        control_mode_index = 1
        arm_trans = np.array([-0.976,0.729,0.087])
        arm_rot = np.array([-0.346,0.602,0.602,-0.396])
        hand_joints = self._hand_init_joints

      elif 5 + 2 * self.time_interval < step <= 5 + 3 * self.time_interval:
        print("=== Phase: 3 ===")
        target_velocity = np.zeros(self.arm_dof+self.hand_dof)
        target_velocity[self.arm_dof:] = 1
        step = (step-(5 + 2 * self.time_interval)) / self.time_interval
        control_mode_index = 1
        arm_trans = np.array([-0.976,0.729,0.087])
        arm_rot = np.array([-0.346,0.602,0.602,-0.396])
        # action hand
        hand_prev_joints = self._hand_init_joints
        hand_next_joints = self.normalize_hand_joints(np.array([0.8567997, 0.0, 0.8441997, 0.0, 0.8882997, 0.072, 0.9008997, 0.125, -2.094, 0.0]))
        hand_joints = self.interpolate_pos(step, hand_prev_joints, hand_next_joints, 0, 1)

      elif 5 + 3 * self.time_interval < step <= 5 + 4 * self.time_interval:
        print("=== Phase: 4 ===")
      
        target_velocity = np.zeros(self.arm_dof+self.hand_dof)
        target_velocity[self.arm_dof:] = 1
        step = (step-(5 + 3 * self.time_interval)) / self.time_interval
        control_mode_index = 1
        arm_trans = np.array([-0.976,0.729,0.087])
        arm_rot = np.array([-0.346,0.602,0.602,-0.396])
        # action hand
        hand_prev_joints = self.normalize_hand_joints(np.array([0.8567997, 0.0, 0.8441997, 0.0, 0.8882997, 0.072, 0.9008997, 0.125, -2.094, 0.0]))
        hand_next_joints = self.normalize_hand_joints(np.array([0.7711194, 1.217, 0.75977945, 1.396, 0.91, 1.083, 0.81080943, 1.146, -1.8845994, 0.536]))
        hand_joints = self.interpolate_pos(step, hand_prev_joints, hand_next_joints, 0, 1)
    # """get close to the plate"""
    

    else:
      print("Wrong step")

    if control_mode_index == 0:
      arm_pose = arm_joints
    elif control_mode_index == 1:  
      arm_pose = np.concatenate((arm_trans, arm_rot))
      
    else:
      raise NotImplementedError
    
    key_point_info = dict(
        arm_pose=arm_pose,
        hand_joints= hand_joints,
        control_mode_index= control_mode_index
      )
    
    return key_point_info

  def get_wash_stage_key_points(self, step):
    pass

  def get_insert_stage_key_points(self, step):
    pass
  
  def set_up_key_point_property(self, step):
    arm_trans, arm_rot, arm_joints, hand_joints, control_mode_index = None, None, None, None, 0
    # arm_trans, arm_rot, hand_joints, control_mode_index = self.get_pickup_stage_key_points(step)
    arm_trans, arm_rot, hand_joints, control_mode_index = self.get_picksponge_stage_key_points(step)
    if control_mode_index == 1:
      arm_pose = np.zeros(7)
      arm_pose[0:3], arm_pose[3:7] = arm_trans, arm_rot
    elif control_mode_index == 2:
      arm_pose = np.zeros(6)
      arm_pose[0:3], arm_pose[3:6] = arm_trans, arm_rot
    elif control_mode_index == 0:
      arm_pose = np.zeros(self.arm_dof)
      arm_pose[0:self.arm_dof] = arm_joints
    else:
      raise NotImplementedError
    
    key_point_info = dict(
        arm_pose=arm_pose,
        hand_joints= hand_joints,
        control_mode_index= control_mode_index
      )
    
    return key_point_info

  

  def interpolate_pos(self, step, pos_start, pos_end, t_start, t_end):
      return pos_start + (pos_end - pos_start) * ((step - t_start) / (t_end - t_start))

  def interpolate_quat(self, step, quat_start, quat_end, t_start, t_end):
      euler_start = R.from_quat(quat_start).as_euler('xyz')
      euler_end = R.from_quat(quat_end).as_euler('xyz')
      euler_interpolated = self.interpolate_pos(step, euler_start, euler_end, t_start, t_end)
      return R.from_euler('xyz', euler_interpolated).as_quat()

  def slerp_quat(self, step, quat_start, quat_end, t_start, t_end):
      # Normalize the quaternions just to be safe
      quat_start = quat_start / np.linalg.norm(quat_start)
      quat_end = quat_end / np.linalg.norm(quat_end)

      # Compute the cosine of the angle between the two vectors.
      dot = np.dot(quat_start, quat_end)

      # If the dot product is negative, slerp won'step take the shorter path.
      # Note that v1 and -v1 are equivalent when the negation is applied to all four components. 
      if dot < 0.0:
          quat_end = -quat_end
          dot = -dot

      DOT_THRESHOLD = 0.9995
      if dot > DOT_THRESHOLD:
          # If the inputs are too close for comfort, linearly interpolate
          # and normalize the result.
          result = quat_start + step * (quat_end - quat_start)
          return result / np.linalg.norm(result)

      # Since dot is in range [0, DOT_THRESHOLD], acos is safe
      theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
      theta = theta_0 * step  # theta = angle between v0 and result
      sin_theta = np.sin(theta)  # compute this value only once
      sin_theta_0 = np.sin(theta_0)  # compute this value only once

      s0 = np.cos(theta) - dot * sin_theta / sin_theta_0  # == sin(theta_0 - theta) / sin(theta_0)
      s1 = sin_theta / sin_theta_0
      return (s0 * quat_start) + (s1 * quat_end)

  def normalize_hand_joints(self, hand_joints):
      qlimit_scope = self._qlimit_scope[self.arm_dof:]
      normalized_joints = np.zeros_like(hand_joints)
      for i, (q_min, q_max) in enumerate(qlimit_scope):
          normalized_joints[i] = 2 * (hand_joints[i] - q_min) / (q_max - q_min) - 1
      return normalized_joints

  def normalize_arm_joints(self, arm_joints):
      qlimit_scope = self._qlimit_scope[0: self.arm_dof]
      normalized_joints = np.zeros_like(arm_joints)
      for i, (q_min, q_max) in enumerate(qlimit_scope):
          normalized_joints[i] = 2 * (arm_joints[i] - q_min) / (q_max - q_min) - 1
      return normalized_joints
  

CAMERA_CONFIG = {
    # ====================================================================
    # pickup -> above_*np.sink -> wash -> above_disk_rack -> insert_disk_rack
    # ====================================================================
    "tidy_up_dishes": 
        {
            "pickup": 
                    {
                        "go to the *np.sink": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "prepare capture": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "capture the plate": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                    },
            "above_*np.sink": 
                    {
                        "arm_pose": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "hand_pose":Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                    },
            "wash": 
                    {
                        "arm_pose": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "hand_pose":Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                    },
            "above_disk_rack": 
                    {
                        "arm_pose": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "hand_pose":Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                    },
            "insert_disk_rack": 
                    {
                        "arm_pose": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "hand_pose":Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                    },
        },

    "drill_screw": 
        {
            "pick_plate": 
                    {
                        "arm_pose": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "hand_pose":Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                    },
            "place_plate": 
                    {
                        "arm_pose": Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                        "hand_pose":Pose(p=[-1.057, 0.03075, 0.006],q=[-0.291, 0, 0, 0.957]),
                    },
        },
}