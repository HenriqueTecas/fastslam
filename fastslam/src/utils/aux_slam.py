import numpy as np
import math
import random
import copy

def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi]
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def cart2pol(x, y):
    """
    Convert Cartesian to polar coordinates
    
    Args:
        x: X coordinate
        y: Y coordinate
        
    Returns:
        (rho, phi) - Distance and angle in degrees
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, math.degrees(phi))

def pol2cart(rho, phi_deg):
    """
    Convert polar to Cartesian coordinates
    
    Args:
        rho: Distance
        phi_deg: Angle in degrees
        
    Returns:
        (x, y) coordinates
    """
    phi = math.radians(phi_deg)
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return (x, y)

def normalize_weights(particles, num_particles):
    """
    Normalize the weights of particles to sum to 1
    
    Args:
        particles: List of particles
        num_particles: Number of particles
        
    Returns:
        Particles with normalized weights
    """
    sumw = sum([p.weight for p in particles])
    try:
        for i in range(num_particles):
            particles[i].weight /= sumw
    except ZeroDivisionError:
        # If sum is zero, set equal weights
        for i in range(num_particles):
            particles[i].weight = 1.0 / num_particles
    return particles

def low_variance_resampling(weights, equal_weights, num_particles):
    """
    Low variance resampling algorithm
    
    Args:
        weights: Current particle weights
        equal_weights: Equal weights (1/N)
        num_particles: Number of particles
        
    Returns:
        Indices of selected particles
    """
    # Compute cumulative sum of weights
    wcum = np.cumsum(weights)
    
    # Create base points
    base = np.cumsum(equal_weights) - 1 / num_particles
    
    # Add random offset
    resampleid = base + np.random.rand(base.shape[0]) / num_particles
    
    # Select particles
    indices = np.zeros(num_particles, dtype=int)
    j = 0
    for i in range(num_particles):
        while j < wcum.shape[0] - 1 and resampleid[i] > wcum[j]:
            j += 1
        indices[i] = j
    
    return indices

def stratified_resampling(weights, num_particles):
    """
    Stratified resampling algorithm
    
    Args:
        weights: Particle weights
        num_particles: Number of particles
        
    Returns:
        Indices of selected particles
    """
    # Compute cumulative sum of weights
    cumulative_weights = np.cumsum(weights)
    
    # Create stratified samples
    strata = np.linspace(0, 1, num_particles + 1)[:-1]
    strata += np.random.rand(num_particles) / num_particles
    
    # Select particles
    indices = np.zeros(num_particles, dtype=int)
    j = 0
    for i in range(num_particles):
        while cumulative_weights[j] < strata[i]:
            j += 1
        indices[i] = j
    
    return indices

def resample(particles, num_particles, resample_method, best_particle_id):
    """
    Resample particles based on weights
    
    Args:
        particles: List of particles
        num_particles: Number of particles
        resample_method: Method for resampling ("low variance" or "stratified")
        best_particle_id: ID of current best particle
        
    Returns:
        Resampled particles and new best particle ID
    """
    # Normalize weights
    particles = normalize_weights(particles, num_particles)
    weights = np.array([particle.weight for particle in particles])
    
    # Find highest weight particle
    highest_weight_index = np.argmax(weights)
    
    # Calculate effective particle number
    Neff = 1.0 / np.sum(np.square(weights))
    equal_weights = np.full_like(weights, 1 / num_particles)
    Neff_maximum = 1.0 / np.sum(np.square(equal_weights))
    
    # New best particle ID after resampling
    new_best_particle_id = highest_weight_index
    
    # Resample if effective particle number is too low
    if Neff < 0.5 * Neff_maximum:
        if resample_method == "low variance":
            indices = low_variance_resampling(weights, equal_weights, num_particles)
        elif resample_method == "stratified":
            indices = stratified_resampling(weights, num_particles)
        else:
            # Default to low variance
            indices = low_variance_resampling(weights, equal_weights, num_particles)
        
        # Create deep copies of selected particles
        particles_copy = copy.deepcopy(particles)
        
        # Replace particles
        for i in range(num_particles):
            particles[i].pose = particles_copy[indices[i]].pose
            particles[i].landmarks = particles_copy[indices[i]].landmarks
            particles[i].weight = 1.0 / num_particles
            
            # Update best particle ID
            if highest_weight_index == indices[i]:
                new_best_particle_id = i
    
    return particles, new_best_particle_id

def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two points
    
    Args:
        a: First point (x, y)
        b: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
