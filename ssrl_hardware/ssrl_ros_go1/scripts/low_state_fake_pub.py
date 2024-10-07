#!/usr/bin/env python
from unitree_legged_msgs.msg import LowState
from brax.robots.go1.utils import Go1Utils 

import rospy
import numpy as np


def publish_low_state():
    rospy.loginfo("Starting fake low state publisher node")

    # Initialize the node
    rospy.init_node('low_state_publisher', anonymous=True)

    # Create a publisher object
    pub = rospy.Publisher('/low_state', LowState, queue_size=10)

    # Set the loop rate (in Hz)
    rate = rospy.Rate(100)

    # Keep publishing until the node is stopped
    while not rospy.is_shutdown():
        # Create a new LowState message
        low_state_msg = LowState()
        low_state_msg.imu.quaternion = [1.0, 0.0, 0.0, 0.0]
        low_state_msg.imu.accelerometer = (np.array([0.0, 0.0, 9.81])
                                           + np.random.uniform(-0.1, 0.1, 3))
        for i in range(12):
            low_state_msg.motorState[i].q = (Go1Utils.ALL_STANDING_JOINT_ANGLES[i] 
                                             + np.random.uniform(-0.05, 0.05))

        # Publish the message
        pub.publish(low_state_msg)

        # Sleep for the remainder of the loop
        rate.sleep()


if __name__ == '__main__':
    try:
        publish_low_state()
    except rospy.ROSInterruptException:
        pass
