import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error
import math

def align_trajectories(estimate, reference):
    """
    Align estimated trajectory to reference trajectory using Horn's method
    
    Args:
        estimate: Estimated trajectory as Nx3 array (x, y, theta)
        reference: Reference trajectory as Nx3 array (x, y, theta)
        
    Returns:
        Aligned estimated trajectory
    """
    # Extract positions (discard orientation)
    est_pos = estimate[:, :2]
    ref_pos = reference[:, :2]
    
    # Calculate centroids
    est_centroid = np.mean(est_pos, axis=0)
    ref_centroid = np.mean(ref_pos, axis=0)
    
    # Center trajectories
    est_centered = est_pos - est_centroid
    ref_centered = ref_pos - ref_centroid
    
    # Calculate cross-covariance matrix
    H = np.dot(est_centered.T, ref_centered)
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Calculate rotation
    R_align = np.dot(Vt.T, U.T)
    
    # Ensure proper rotation matrix (no reflection)
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = np.dot(Vt.T, U.T)
    
    # Calculate translation
    t_align = ref_centroid - np.dot(R_align, est_centroid)
    
    # Apply transformation to estimated trajectory
    aligned_est = np.zeros_like(estimate)
    for i in range(len(estimate)):
        # Apply rotation and translation to position
        aligned_est[i, :2] = np.dot(R_align, est_pos[i]) + t_align
        
        # Apply rotation to orientation
        r_est = R.from_euler('z', estimate[i, 2])
        r_align = R.from_matrix(np.vstack([np.hstack([R_align, np.zeros((2, 1))]),
                                          [0, 0, 1]]))
        r_new = r_align * r_est
        aligned_est[i, 2] = r_new.as_euler('zyx')[0]
    
    return aligned_est

def absolute_trajectory_error(estimate, reference):
    """
    Calculate Absolute Trajectory Error (ATE)
    
    Args:
        estimate: Estimated trajectory as Nx3 array (x, y, theta)
        reference: Reference trajectory as Nx3 array (x, y, theta)
        
    Returns:
        ATE as RMSE
    """
    # Align trajectories
    aligned_est = align_trajectories(estimate, reference)
    
    # Calculate position error
    pos_error = np.linalg.norm(aligned_est[:, :2] - reference[:, :2], axis=1)
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean(np.square(pos_error)))
    
    return rmse

def relative_pose_error(estimate, reference, delta=10):
    """
    Calculate Relative Pose Error (RPE)
    
    Args:
        estimate: Estimated trajectory as Nx3 array (x, y, theta)
        reference: Reference trajectory as Nx3 array (x, y, theta)
        delta: Number of frames to compare
        
    Returns:
        Translational and rotational RPE
    """
    n = len(estimate)
    trans_errors = []
    rot_errors = []
    
    for i in range(0, n - delta):
        # Calculate relative motion in estimated trajectory
        est_delta_trans = np.linalg.norm(estimate[i + delta, :2] - estimate[i, :2])
        est_delta_rot = normalize_angle(estimate[i + delta, 2] - estimate[i, 2])
        
        # Calculate relative motion in reference trajectory
        ref_delta_trans = np.linalg.norm(reference[i + delta, :2] - reference[i, :2])
        ref_delta_rot = normalize_angle(reference[i + delta, 2] - reference[i, 2])
        
        # Calculate errors
        trans_errors.append(abs(est_delta_trans - ref_delta_trans))
        rot_errors.append(abs(normalize_angle(est_delta_rot - ref_delta_rot)))
    
    # Calculate RMSE
    trans_rmse = np.sqrt(np.mean(np.square(trans_errors)))
    rot_rmse = np.sqrt(np.mean(np.square(rot_errors)))
    
    return trans_rmse, rot_rmse

def landmark_position_error(estimated_landmarks, true_landmarks):
    """
    Calculate error in landmark positions
    
    Args:
        estimated_landmarks: Dictionary of estimated landmarks {id: landmark}
        true_landmarks: Dictionary of true landmarks {id: (x, y)}
        
    Returns:
        RMSE of landmark positions
    """
    errors = []
    
    # Check each landmark
    for lm_id, true_pos in true_landmarks.items():
        if lm_id in estimated_landmarks:
            est_pos = [estimated_landmarks[lm_id].x, estimated_landmarks[lm_id].y]
            error = np.linalg.norm(np.array(est_pos) - np.array(true_pos))
            errors.append(error)
    
    # Calculate RMSE
    if len(errors) > 0:
        rmse = np.sqrt(np.mean(np.square(errors)))
        return rmse
    else:
        return float('inf')  # No common landmarks

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def evaluate_slam_performance(estimated_trajectory, ground_truth_trajectory, 
                             estimated_landmarks=None, true_landmarks=None):
    """
    Comprehensive evaluation of SLAM performance
    
    Args:
        estimated_trajectory: List of estimated poses (x, y, theta)
        ground_truth_trajectory: List of ground truth poses
        estimated_landmarks: Dictionary of estimated landmarks
        true_landmarks: Dictionary of true landmarks
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert to numpy arrays
    est_traj = np.array(estimated_trajectory)
    gt_traj = np.array(ground_truth_trajectory)
    
    # Ensure same length
    min_len = min(len(est_traj), len(gt_traj))
    est_traj = est_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    
    # Calculate metrics
    results = {}
    
    # Trajectory metrics
    results['ate'] = absolute_trajectory_error(est_traj, gt_traj)
    results['rpe_trans'], results['rpe_rot'] = relative_pose_error(est_traj, gt_traj)
    
    # Landmark metrics (if available)
    if estimated_landmarks is not None and true_landmarks is not None:
        results['landmark_rmse'] = landmark_position_error(estimated_landmarks, true_landmarks)
    
    return results
