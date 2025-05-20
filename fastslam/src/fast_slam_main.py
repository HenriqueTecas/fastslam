#!/usr/bin/env python3

import rospy
import argparse
import os
from PioneerFastSLAM import PioneerFastSLAM

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pioneer FastSLAM with ArUco Markers")
    parser.add_argument('--num_particles', type=int, default=50, help="Number of particles")
    parser.add_argument('--record_data', action='store_true', help="Record data for later analysis")
    parser.add_argument('--playback', type=str, default="", help="Playback recorded rosbag file")
    parser.add_argument('--map_size', type=float, default=10.0, help="Size of map in meters")
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('pioneer_fastslam')
    
    # Set up SLAM parameters
    wheel_base = 0.38  # Pioneer 3DX wheel base in meters
    
    # Motion model parameters
    alphas = [0.00001, 0.00001, 0.00001, 0.00001]  # Motion model noise parameters
    
    # Initialize SLAM
    slam = PioneerFastSLAM(
        num_particles=args.num_particles,
        wheel_base=wheel_base,
        alphas=alphas,
        map_size=args.map_size,
        record_data=args.record_data
    )
    
    # Start SLAM processing
    if args.playback:
        if not os.path.exists(args.playback):
            rospy.logerr(f"Rosbag file {args.playback} does not exist!")
            return
        rospy.loginfo(f"Playing back rosbag file: {args.playback}")
        slam.run_with_rosbag(args.playback)
    else:
        rospy.loginfo("Starting live SLAM...")
        slam.run()
    
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
