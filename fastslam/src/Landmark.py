import numpy as np
import math

class Landmark:
    """
    Represents a landmark in the FastSLAM map
    
    Attributes:
        x: X-coordinate of the landmark
        y: Y-coordinate of the landmark
        sigma: 3x3 covariance matrix representing uncertainty
    """
    def __init__(self, x, y):
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate
        
        # Covariance matrix (3x3) for x, y, theta
        # Initialize with high uncertainty
        self.sigma = np.eye(3) * 1000.0
        
        # Additional fields for tracking observations
        self.observation_count = 1
        self.last_observed = 0  # timestamp
    
    def update_position(self, x, y):
        """Update the landmark position"""
        self.x = x
        self.y = y
        self.observation_count += 1
    
    def distance_to(self, x, y):
        """Calculate distance to a point"""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def update_last_observed(self, timestamp):
        """Update when landmark was last observed"""
        self.last_observed = timestamp
    
    def get_position(self):
        """Get position as numpy array"""
        return np.array([self.x, self.y])
