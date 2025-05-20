import cv2
import numpy as np

def detect_aruco_markers(image, aruco_dict, parameters, camera_matrix, dist_coeffs, marker_length=0.1):
    """
    Detect ArUco markers in an image
    
    Args:
        image: Input image
        aruco_dict: ArUco dictionary
        parameters: ArUco detector parameters
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        marker_length: Physical size of marker in meters
        
    Returns:
        markers: List of detected markers with format (id, distance, angle)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    
    markers = []
    
    if ids is not None and len(ids) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        # Estimate pose for each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)
        
        for i in range(len(ids)):
            # Draw axes for each marker
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, 
                           rvecs[i], tvecs[i], marker_length/2)
            
            # Get position in camera frame
            tvec = tvecs[i][0]
            
            # Transform to robot frame (this will be specific to your setup)
            tvec_robot = transform_camera_to_robot(tvec)
            
            # Calculate distance and angle
            distance = np.linalg.norm(tvec_robot[:2])
            angle = np.arctan2(tvec_robot[1], tvec_robot[0])
            
            markers.append((ids[i][0], distance, angle))
            
            # Add text with marker info
            cv2.putText(image, f"ID: {ids[i][0]} D: {distance:.2f}m", 
                       (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, markers

def transform_camera_to_robot(tvec):
    """
    Transform from camera coordinates to robot coordinates
    
    Args:
        tvec: Translation vector in camera frame
        
    Returns:
        tvec_robot: Translation vector in robot frame
    """
    # This transformation will be specific to your robot setup
    # For Pioneer 3DX, you'll need to measure/calibrate these values
    
    # Example transformation assuming camera mounted at (0.1, 0, 0.3) with identity rotation
    R_cam_to_robot = np.eye(3)  # No rotation
    t_cam_to_robot = np.array([0.1, 0, 0.3])  # Camera position relative to robot center
    
    # Apply transformation
    tvec_robot = R_cam_to_robot @ tvec + t_cam_to_robot
    
    return tvec_robot
