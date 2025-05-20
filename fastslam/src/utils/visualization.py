import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import pygame

def draw_robot(screen, robot_pose, color, radius, screen_width, screen_height, width_meters, height_meters):
    """
    Draw a robot on the screen
    
    Args:
        screen: Pygame screen
        robot_pose: (x, y, theta) - Robot pose
        color: RGB color tuple
        radius: Robot radius in pixels
        screen_width: Width of screen in pixels
        screen_height: Height of screen in pixels
        width_meters: Width of environment in meters
        height_meters: Height of environment in meters
    """
    x, y, theta = robot_pose
    
    # Convert to pixel coordinates
    pixel_x = int(x * screen_width / width_meters + screen_width / 2)
    pixel_y = int(y * screen_height / height_meters + screen_height / 2)
    
    # Draw robot body
    pygame.draw.circle(screen, color, (pixel_x, pixel_y), radius)
    
    # Draw orientation indicator
    end_x = pixel_x + radius * math.cos(theta)
    end_y = pixel_y - radius * math.sin(theta)
    pygame.draw.line(screen, (0, 0, 0), (pixel_x, pixel_y), (end_x, end_y), 2)

def draw_landmark(screen, landmark_pos, color, size, screen_width, screen_height, width_meters, height_meters):
    """
    Draw a landmark on the screen
    
    Args:
        screen: Pygame screen
        landmark_pos: (x, y) - Landmark position
        color: RGB color tuple
        size: Size of landmark in pixels
        screen_width: Width of screen in pixels
        screen_height: Height of screen in pixels
        width_meters: Width of environment in meters
        height_meters: Height of environment in meters
    """
    x, y = landmark_pos
    
    # Convert to pixel coordinates
    pixel_x = int(x * screen_width / width_meters + screen_width / 2)
    pixel_y = int(y * screen_height / height_meters + screen_height / 2)
    
    # Draw landmark
    pygame.draw.circle(screen, color, (pixel_x, pixel_y), size)

def draw_particles(screen, particles, color, screen_width, screen_height, width_meters, height_meters):
    """
    Draw particles on the screen
    
    Args:
        screen: Pygame screen
        particles: List of particles
        color: RGB color tuple
        screen_width: Width of screen in pixels
        screen_height: Height of screen in pixels
        width_meters: Width of environment in meters
        height_meters: Height of environment in meters
    """
    for particle in particles:
        x, y, _ = particle.pose
        
        # Convert to pixel coordinates
        pixel_x = int(x * screen_width / width_meters + screen_width / 2)
        pixel_y = int(y * screen_height / height_meters + screen_height / 2)
        
        # Size proportional to weight
        size = max(1, int(3 * particle.weight * len(particles)))
        
        # Draw particle
        pygame.draw.circle(screen, color, (pixel_x, pixel_y), size)

def display_aruco_markers(image, corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs, marker_length=0.1):
    """
    Display detected ArUco markers on image
    
    Args:
        image: Input image
        corners: Corners of detected markers
        ids: IDs of detected markers
        rvecs: Rotation vectors
        tvecs: Translation vectors
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        marker_length: Size of marker in meters
        
    Returns:
        Image with visualized markers
    """
    output_image = image.copy()
    
    # Draw detected markers
    cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
    
    # Draw axes for each marker
    if rvecs is not None and tvecs is not None:
        for i in range(len(ids)):
            cv2.drawFrameAxes(output_image, camera_matrix, dist_coeffs, 
                              rvecs[i], tvecs[i], marker_length/2)
            
            # Add text with ID and distance
            tvec = tvecs[i][0]
            distance = np.linalg.norm(tvec)
            text = f"ID: {ids[i][0]}, D: {distance:.2f}m"
            
            # Get corner position for text
            corner = corners[i][0][0]
            cv2.putText(output_image, text, 
                        (int(corner[0]), int(corner[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_image

def plot_map_and_trajectory(landmarks, trajectory, ground_truth=None, save_path=None):
    """
    Plot map and trajectory
    
    Args:
        landmarks: Dictionary of landmarks {id: landmark}
        trajectory: List of (x, y, theta) poses
        ground_truth: Optional ground truth trajectory
        save_path: Path to save the plot (None for display)
    """
    plt.figure(figsize=(10, 8))
    
    # Plot landmarks
    lm_x = []
    lm_y = []
    lm_ids = []
    
    for lm_id, landmark in landmarks.items():
        lm_x.append(landmark.x)
        lm_y.append(landmark.y)
        lm_ids.append(lm_id)
    
    plt.scatter(lm_x, lm_y, c='blue', marker='s', s=100, label='Landmarks')
    
    # Add landmark IDs
    for i, lm_id in enumerate(lm_ids):
        plt.annotate(lm_id, (lm_x[i], lm_y[i]), fontsize=8)
    
    # Plot estimated trajectory
    traj_x = [pose[0] for pose in trajectory]
    traj_y = [pose[1] for pose in trajectory]
    plt.plot(traj_x, traj_y, 'r-', linewidth=2, label='Estimated Trajectory')
    
    # Plot ground truth if available
    if ground_truth is not None:
        gt_x = [pose[0] for pose in ground_truth]
        gt_y = [pose[1] for pose in ground_truth]
        plt.plot(gt_x, gt_y, 'g--', linewidth=2, label='Ground Truth')
    
    # Mark start and end
    plt.plot(traj_x[0], traj_y[0], 'ro', markersize=10, label='Start')
    plt.plot(traj_x[-1], traj_y[-1], 'rx', markersize=10, label='End')
    
    # Add grid and labels
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('FastSLAM Map and Trajectory')
    plt.legend()
    plt.axis('equal')
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
