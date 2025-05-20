#!/usr/bin/env python3

import rospy
import numpy as np
import tf
import cv2
import threading
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

from FastSLAM import FastSLAM
from utils.aux_slam import normalize_angle, cart2pol
import subprocess

class PioneerFastSLAM:
    def __init__(self, num_particles=50, wheel_base=0.38, alphas=None, map_size=10.0, record_data=False):
        # Initialize parameters
        self.num_particles = num_particles
        self.wheel_base = wheel_base
        self.map_size = map_size
        self.record_data = record_data
        
        # Initialize ROS-related fields
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # Set up ArUco detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Initialize SLAM algorithm
        tuning_options = [
            np.diag([0.1, 0.1]),  # Q_init: Initial measurement noise
            np.diag([0.7, 0.7]),  # Q_update: Update measurement noise
            alphas if alphas else [0.00001, 0.00001, 0.00001, 0.00001]  # Motion model noise
        ]
        
        # Initialize the FastSLAM algorithm
        self.slam = FastSLAM(
            window_size_pixel=800,
            size_m=map_size,
            central_bar_width=10,
            wheel_base=wheel_base,
            motion_model_type='differential_drive',
            num_particles=num_particles,
            tuning_options=tuning_options
        )
        
        # Initial state
        self.current_aruco = []
        self.odometry = [0, 0, 0]  # [x, y, theta]
        self.old_odometry = [0, 0, 0]
        self.first_reading = True
        
        # Initialize subscribers
        self.odom_sub = rospy.Subscriber('/pose', Odometry, self.odom_callback)
        
        # Try first the compressed image topic, then fall back to raw
        try:
            self.image_sub = rospy.Subscriber('/camera/image/compressed', 
                                              CompressedImage, 
                                              self.compressed_image_callback)
            rospy.loginfo("Subscribed to compressed image topic")
        except:
            self.image_sub = rospy.Subscriber('/camera/image', 
                                              Image, 
                                              self.image_callback)
            rospy.loginfo("Subscribed to raw image topic")
        
        # Publishers
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        
        # Initialize camera calibration
        self.calibrate_camera()
        
        # Recording setup
        if record_data:
            self.setup_recording()
            
        rospy.loginfo("Pioneer FastSLAM initialized with %d particles", num_particles)
    
    def calibrate_camera(self):
        """Set up camera calibration parameters"""
        # These are default parameters, replace with your calibrated values
        fx, fy = 565.6, 565.6  # Focal lengths
        cx, cy = 320.5, 240.5  # Principal point
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Distortion coefficients [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    def setup_recording(self):
        """Set up data recording using rosbag"""
        timestamp = rospy.Time.now().to_sec()
        filename = f"pioneer_slam_{timestamp}.bag"
        topics = ["/pose", "/camera/image/compressed", "/landmarks"]
        
        # Start rosbag record process
        cmd = ["rosbag", "record", "-O", filename] + topics
        self.rosbag_proc = subprocess.Popen(cmd)
        rospy.loginfo(f"Recording data to {filename}")
    
    def odom_callback(self, data):
        """Process odometry data"""
        with self.lock:
            # Extract position and orientation from odometry
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            
            # Extract orientation quaternion
            quaternion = [
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w
            ]
            
            # Convert quaternion to Euler angles
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
            
            # Update odometry
            self.odometry = [x, y, yaw]
            
            # Initialize reference position on first callback
            if self.first_reading:
                self.old_odometry = [x, y, yaw]
                self.first_reading = False
                return
            
            # Calculate odometry change
            delta_dist = np.sqrt((x - self.old_odometry[0])**2 + (y - self.old_odometry[1])**2)
            delta_rot1 = normalize_angle(np.arctan2(y - self.old_odometry[1], 
                                                   x - self.old_odometry[0]) - self.old_odometry[2])
            delta_rot2 = normalize_angle(yaw - self.old_odometry[2] - delta_rot1)
            
            # Update SLAM with odometry
            self.slam.update_odometry([delta_dist, delta_rot1, delta_rot2])
            
            # Update old odometry
            self.old_odometry = [x, y, yaw]
            
            # Update visualization
            self.publish_tf()
    
    def compressed_image_callback(self, data):
        """Process compressed camera images"""
        with self.lock:
            try:
                # Convert compressed image to OpenCV format
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
                self.process_image(cv_image)
            except Exception as e:
                rospy.logerr(f"Error processing compressed image: {e}")
    
    def image_callback(self, data):
        """Process raw camera images"""
        with self.lock:
            try:
                # Convert image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.process_image(cv_image)
            except Exception as e:
                rospy.logerr(f"Error processing image: {e}")
    
    def process_image(self, cv_image):
        """Process image to detect ArUco markers"""
        self.current_aruco = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers on the image
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            for i in range(len(ids)):
                # Estimate pose of marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], 0.25, self.camera_matrix, self.dist_coeffs)
                
                # Transform marker position to robot coordinates
                tvec = self.transform_camera_to_robot(tvecs[0][0])
                
                # Convert to polar coordinates (distance, angle)
                dist, phi = cart2pol(tvec[0], tvec[2])
                
                # Add detected marker to the current observations
                self.current_aruco.append((dist, -phi, ids[i][0]))
                
                # Draw marker information on the image
                cv2.putText(cv_image, f"ID: {ids[i][0]} D: {dist:.2f}m", 
                           (int(corners[i][0][0][0]), int(corners[i][0][0][1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Update SLAM with the detected markers
        self.slam.compute_slam(self.current_aruco)
        
        # Display the image
        cv2.imshow("ArUco Detection", cv_image)
        cv2.waitKey(1)
    
    def transform_camera_to_robot(self, tvec):
        """Transform coordinates from camera frame to robot frame"""
        # Rotation matrix from camera to robot
        R_cam_to_robot = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Translation from camera to robot center (adjust these values)
        T_cam_to_robot = np.array([0.076, 0, 0.103])
        
        # Homogeneous transformation
        tvec_hom = np.append(tvec, [1])
        
        # Create transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_cam_to_robot
        transform_matrix[:3, 3] = T_cam_to_robot
        
        # Apply transformation
        tvec_robot_hom = np.dot(transform_matrix, tvec_hom)
        
        return tvec_robot_hom[:3]
    
    def publish_tf(self):
        """Publish TF frames for visualization"""
        # Get best particle
        best_particle = self.slam.get_best_particle()
        x, y, theta = best_particle.pose
        
        # Broadcast map to odom transform
        self.tf_broadcaster.sendTransform(
            (0.0, 0.0, 0.0),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            "odom",
            "map"
        )
        
        # Broadcast odom to base_link transform
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta - np.pi)
        self.tf_broadcaster.sendTransform(
            (-x, y, 0),
            quaternion,
            rospy.Time.now(),
            "base_link",
            "odom"
        )
        
        # Publish landmark markers
        self.publish_landmarks()
    
    def publish_landmarks(self):
        """Publish landmark markers for visualization"""
        if not hasattr(self.slam, 'get_best_particle'):
            return
            
        best_particle = self.slam.get_best_particle()
        if not best_particle:
            return
            
        marker_array = MarkerArray()
        
        for landmark_id, landmark in best_particle.landmarks.items():
            # Create a marker for the landmark
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "landmarks"
            marker.id = int(landmark_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = -landmark.x
            marker.pose.position.y = landmark.y
            marker.pose.position.z = 0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
            
            # Add marker ID as text
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "landmark_text"
            text_marker.id = int(landmark_id) + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = -landmark.x
            text_marker.pose.position.y = landmark.y
            text_marker.pose.position.z = 0.3
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.2
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.text = f"ID: {landmark_id}"
            
            marker_array.markers.append(text_marker)
        
        self.landmark_pub.publish(marker_array)
    
    def run(self):
        """Run the SLAM system in live mode"""
        rate = rospy.Rate(10)  # 10 Hz
        
        try:
            while not rospy.is_shutdown():
                # Main processing is done in callbacks
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down...")
        finally:
            # Clean up
            cv2.destroyAllWindows()
            if self.record_data and hasattr(self, 'rosbag_proc'):
                self.rosbag_proc.terminate()
                rospy.loginfo("Stopped recording")
    
    def run_with_rosbag(self, rosbag_file):
        """Run SLAM with a recorded rosbag file"""
        # Start rosbag playback in a separate process
        rosbag_cmd = ["rosbag", "play", rosbag_file]
        rosbag_proc = subprocess.Popen(rosbag_cmd)
        
        # Run SLAM
        self.run()
        
        # Ensure rosbag process is terminated
        rosbag_proc.terminate()
