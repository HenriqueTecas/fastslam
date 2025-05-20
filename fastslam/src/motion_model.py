import numpy as np
import math
from utils.aux_slam import normalize_angle

def differential_drive_model(pose, odometry_delta, wheel_base, alphas):
    """
    Pioneer 3DX differential drive motion model
    
    Args:
        pose: Current robot pose (x, y, theta)
        odometry_delta: [delta_dist, delta_rot1, delta_rot2]
        wheel_base: Distance between wheels
        alphas: Motion model noise parameters [α1, α2, α3, α4]
        
    Returns:
        new_pose: Updated robot pose with noise
    """
    x, y, theta = pose
    delta_dist, delta_rot1, delta_rot2 = odometry_delta
    
    # Noise parameters
    alpha1, alpha2, alpha3, alpha4 = alphas
    
    # Calculate noise stds based on motion
    sigma_rot1 = math.sqrt(alpha1 * delta_rot1**2 + alpha2 * delta_dist**2)
    sigma_trans = math.sqrt(alpha3 * delta_dist**2 + alpha4 * (delta_rot1**2 + delta_rot2**2))
    sigma_rot2 = math.sqrt(alpha1 * delta_rot2**2 + alpha2 * delta_dist**2)
    
    # Sample noise
    delta_rot1_hat = delta_rot1 - np.random.normal(0, sigma_rot1)
    delta_dist_hat = delta_dist - np.random.normal(0, sigma_trans)
    delta_rot2_hat = delta_rot2 - np.random.normal(0, sigma_rot2)
    
    # Apply noisy motion
    new_x = x + delta_dist_hat * math.cos(theta + delta_rot1_hat)
    new_y = y + delta_dist_hat * math.sin(theta + delta_rot1_hat)
    new_theta = normalize_angle(theta + delta_rot1_hat + delta_rot2_hat)
    
    return np.array([new_x, new_y, new_theta])

def compute_odometry_change(x1, y1, theta1, x2, y2, theta2):
    """
    Compute the relative motion between two poses
    
    Args:
        x1, y1, theta1: First pose
        x2, y2, theta2: Second pose
        
    Returns:
        delta_dist, delta_rot1, delta_rot2: Relative motion
    """
    delta_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    delta_rot1 = normalize_angle(math.atan2(y2 - y1, x2 - x1) - theta1)
    delta_rot2 = normalize_angle(theta2 - theta1 - delta_rot1)
    
    return delta_dist, delta_rot1, delta_rot2
