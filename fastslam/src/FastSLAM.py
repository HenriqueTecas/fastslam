#!/usr/bin/env python3

import pygame
import numpy as np
import math
import copy
from Particle import Particle
from utils.aux_slam import resample, normalize_angle, normalize_weights
import rospy
import tf
from visualization_msgs.msg import Marker, MarkerArray

class FastSLAM:
    """
    Implementation of the FastSLAM algorithm for Pioneer 3DX with ArUco markers
    """
    def __init__(self, window_size_pixel, size_m, central_bar_width, wheel_base, 
                motion_model_type='differential_drive', num_particles=50, 
                tuning_options=None, screen=None):
        # Initialize visualization parameters
        self.SCREEN_WIDTH = window_size_pixel
        self.SCREEN_HEIGHT = window_size_pixel
        self.central_bar_width = central_bar_width
        self.width_meters = size_m
        self.height_meters = size_m
        self.robot_radius = 0.2  # Pioneer 3DX radius in meters
        self.robot_radius_pixel = self.robot_radius * self.SCREEN_WIDTH / self.width_meters
        
        # Set up pygame display if not provided
        if screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Pioneer 3DX FastSLAM")
            self.left_coordinate = 0
        else:
            self.screen = screen
            self.left_coordinate = central_bar_width + self.SCREEN_WIDTH
        
        # Define colors for visualization
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 160, 0)
        self.BLUE = (10, 10, 255)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (100, 100, 100)
        
        # FastSLAM algorithm parameters
        self.num_particles = num_particles
        self.wheel_base = wheel_base
        self.motion_model_type = motion_model_type
        self.tuning_options = tuning_options
        self.resample_method = "low variance"  # Resampling method
        
        # Initialize particles
        self.particles = self.initialize_particles()
        self.best_particle_ID = 0  # Initially pick the first particle
        
        # Initialize odometry tracking
        self.old_odometry = [0.0, 0.0]
        self.old_yaw = 0.0
        
        # Draw initial state
        self.update_screen()
    
    def initialize_particles(self):
        """
        Initialize particles with random poses and empty landmark maps
        """
        particles = []
        for _ in range(self.num_particles):
            # Start all particles at origin with small random perturbations
            x = np.random.normal(0, 0.1)
            y = np.random.normal(0, 0.1)
            theta = np.random.normal(0, 0.1)
            pose = np.array([x, y, theta])
            
            # Create a new particle
            particles.append(Particle(pose, self.num_particles, 
                                     self.wheel_base, 
                                     self.motion_model_type, 
                                     self.tuning_options))
        return particles
    
    def update_odometry(self, odometry_delta):
        """
        Update particles based on odometry change
        
        Args:
            odometry_delta: [delta_dist, delta_rot1, delta_rot2]
        """
        # Update each particle with the motion model
        for particle in self.particles:
            particle.motion_model(odometry_delta)
        
        # Update visualization
        self.update_screen()
    
    def compute_slam(self, landmarks_in_sight):
        """
        Perform FastSLAM update with observed landmarks
        
        Args:
            landmarks_in_sight: List of tuples (distance, angle, landmark_id)
        """
        if not landmarks_in_sight:
            return  # No landmarks detected
            
        # Update each particle's landmarks
        for landmark in landmarks_in_sight:
            landmark_dist, landmark_bearing_angle, landmark_id = landmark
            
            # Process this landmark observation in each particle
            for particle in self.particles:
                particle.handle_landmark(landmark_dist, landmark_bearing_angle, landmark_id)
        
        # Resample particles based on weights
        self.particles, self.best_particle_ID = resample(
            self.particles, 
            self.num_particles, 
            self.resample_method, 
            self.best_particle_ID
        )
        
        # Update visualization
        self.update_screen(landmarks_in_sight)
    
    def update_screen(self, landmarks_in_sight=None):
        """
        Update the visualization screen
        
        Args:
            landmarks_in_sight: Currently visible landmarks
        """
        # Clear the screen
        self.screen.fill(self.WHITE)
        
        # Get the best particle (highest weight)
        best_particle = self.get_best_particle()
        x, y, theta = best_particle.pose
        
        # Calculate robot position in pixels
        robot_pos = (
            int(x * self.SCREEN_WIDTH / self.width_meters + self.SCREEN_WIDTH / 2),
            int(y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
        )
        
        # Draw the robot
        pygame.draw.circle(self.screen, self.GREEN, robot_pos, self.robot_radius_pixel)
        
        # Draw orientation indicator (triangle)
        triangle_length = 0.8 * self.robot_radius_pixel
        triangle_tip_x = robot_pos[0] + triangle_length * math.cos(theta)
        triangle_tip_y = robot_pos[1] - triangle_length * math.sin(theta)
        triangle_left_x = robot_pos[0] + triangle_length * math.cos(theta + 5 * math.pi / 6) 
        triangle_left_y = robot_pos[1] - triangle_length * math.sin(theta + 5 * math.pi / 6) 
        triangle_right_x = robot_pos[0] + triangle_length * math.cos(theta - 5 * math.pi / 6)
        triangle_right_y = robot_pos[1] - triangle_length * math.sin(theta - 5 * math.pi / 6)
        
        triangle_points = [
            (triangle_tip_x, triangle_tip_y), 
            (triangle_left_x, triangle_left_y), 
            (triangle_right_x, triangle_right_y)
        ]
        pygame.draw.polygon(self.screen, self.BLUE, triangle_points)
        
        # Draw all particles (red dots)
        for particle in self.particles:
            p_x, p_y, _ = particle.pose
            p_pos = (
                int(p_x * self.SCREEN_WIDTH / self.width_meters + self.SCREEN_WIDTH / 2),
                int(p_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
            )
            # Draw particle with size proportional to weight
            size = max(1, int(3 * particle.weight * self.num_particles))
            pygame.draw.circle(self.screen, self.RED, p_pos, size)
        
        # Draw landmarks from the best particle
        for landmark_id, landmark in best_particle.landmarks.items():
            # Convert landmark position to pixels
            lm_x, lm_y = landmark.x, landmark.y
            lm_pos = (
                int(lm_x * self.SCREEN_WIDTH / self.width_meters + self.SCREEN_WIDTH / 2),
                int(lm_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
            )
            
            # Draw landmark as a black circle
            pygame.draw.circle(self.screen, self.BLACK, lm_pos, 5)
            
            # Add landmark ID text
            font = pygame.font.Font(None, 24)
            id_text = font.render(str(landmark_id), True, self.BLACK)
            self.screen.blit(id_text, (lm_pos[0] + 10, lm_pos[1] - 10))
            
            # Draw a line from robot to landmark
            pygame.draw.line(self.screen, self.GRAY, robot_pos, lm_pos, 1)
        
        # Highlight currently visible landmarks
        if landmarks_in_sight:
            for landmark in landmarks_in_sight:
                _, _, landmark_id = landmark
                
                # Check if this landmark is in the best particle's map
                if str(landmark_id) in best_particle.landmarks:
                    landmark = best_particle.landmarks[str(landmark_id)]
                    lm_x, lm_y = landmark.x, landmark.y
                    lm_pos = (
                        int(lm_x * self.SCREEN_WIDTH / self.width_meters + self.SCREEN_WIDTH / 2),
                        int(lm_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
                    )
                    
                    # Draw a yellow circle around visible landmarks
                    pygame.draw.circle(self.screen, self.YELLOW, lm_pos, 8, 2)
        
        # Draw coordinate axes
        origin = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        pygame.draw.line(self.screen, self.BLACK, origin, 
                        (origin[0] + 50, origin[1]), 2)  # X axis
        pygame.draw.line(self.screen, self.BLACK, origin, 
                        (origin[0], origin[1] - 50), 2)  # Y axis
        
        # Add text labels for axes
        font = pygame.font.Font(None, 20)
        x_label = font.render("X", True, self.BLACK)
        y_label = font.render("Y", True, self.BLACK)
        self.screen.blit(x_label, (origin[0] + 55, origin[1] + 5))
        self.screen.blit(y_label, (origin[0] - 15, origin[1] - 55))
        
        # Add information panel
        self.draw_info_panel()
        
        # Update the display
        pygame.display.flip()
    
    def draw_info_panel(self):
        """Draw information panel with stats"""
        # Create panel background
        panel_rect = pygame.Rect(10, 10, 200, 180)
        pygame.draw.rect(self.screen, (230, 230, 230), panel_rect)
        pygame.draw.rect(self.screen, self.BLACK, panel_rect, 1)
        
        # Get best particle for info
        best_particle = self.get_best_particle()
        
        # Set up font
        font = pygame.font.Font(None, 22)
        y_pos = 15
        line_height = 25
        
        # Add text elements
        texts = [
            f"Particles: {self.num_particles}",
            f"Position: ({best_particle.pose[0]:.2f}, {best_particle.pose[1]:.2f})",
            f"Heading: {math.degrees(best_particle.pose[2]):.1f}°",
            f"Landmarks: {len(best_particle.landmarks)}",
            f"Best ID: {self.best_particle_ID}",
            f"FastSLAM Visualization"
        ]
        
        for text in texts:
            text_surface = font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (20, y_pos))
            y_pos += line_height
    
    def get_best_particle(self):
        """Get the particle with highest weight"""
        return self.particles[self.best_particle_ID]
    
    def get_best_particle_pose(self):
        """Get the pose of the best particle"""
        return self.particles[self.best_particle_ID].pose
    
    def get_map_estimate(self):
        """Get the map estimate from the best particle"""
        return self.particles[self.best_particle_ID].landmarks
