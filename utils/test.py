import numpy as np
from transforms3d.euler import quat2euler

# Initial and final quaternions
quat_start = np.array([-0.137, -0.238, -0.643, 0.715])
quat_end = np.array([-0.448, -0.509, -0.540, 0.499])

# Time steps to evaluate
steps = [0.1, 0.3, 0.5, 0.7, 0.9, 1]



def slerp_quat(t, quat_start, quat_end, t_start, t_end):
    # Normalize the quaternions just to be safe
    quat_start = quat_start / np.linalg.norm(quat_start)
    quat_end = quat_end / np.linalg.norm(quat_end)

    # Compute the cosine of the angle between the two vectors.
    dot = np.dot(quat_start, quat_end)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components. 
    if dot < 0.0:
        quat_end = -quat_end
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate
        # and normalize the result.
        result = quat_start + t * (quat_end - quat_start)
        return result / np.linalg.norm(result)

    # Since dot is in range [0, DOT_THRESHOLD], acos is safe
    theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t  # theta = angle between v0 and result
    sin_theta = np.sin(theta)  # compute this value only once
    sin_theta_0 = np.sin(theta_0)  # compute this value only once

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0  # == sin(theta_0 - theta) / sin(theta_0)
    s1 = sin_theta / sin_theta_0
    return (s0 * quat_start) + (s1 * quat_end)

for t in steps:
    arm_rotation = slerp_quat(t, quat_start, quat_end, 0, 1)
    print(f"t = {t}: {arm_rotation}")


point = quat2euler([0.525861, 0.111989, 0.0700872, -0.840248])
print(point)


import imageio.v2 as imageio
import os
image_folder = 'images'  
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  
image_files = [imageio.imread(os.path.join(image_folder, img)) for img in images]
imageio.mimsave('pickup_sponge.gif', image_files, duration=1000/10) 