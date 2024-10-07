#!/usr/bin/env python
from geometry_msgs.msg import TwistStamped
import rospy
import numpy as np


def publish_twist():
    rospy.loginfo("Starting fake vicon publisher node")

    # Initialize the node
    rospy.init_node('vicon_publisher')

    # Create a publisher object
    pub = rospy.Publisher('vrpn_client_node/quad/twist', TwistStamped,
                          queue_size=10)

    # Set the loop rate (in Hz)
    rate = rospy.Rate(100)

    # Keep publishing until the node is stopped
    while not rospy.is_shutdown():
        # Create a new TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.twist.linear.x = np.random.uniform(-0.05, 0.05)
        twist_msg.twist.linear.y = np.random.uniform(-0.05, 0.05)
        twist_msg.twist.linear.z = np.random.uniform(-0.05, 0.05)

        # Publish the message
        pub.publish(twist_msg)

        # Sleep for the remainder of the loop
        rate.sleep()


if __name__ == '__main__':
    try:
        publish_twist()
    except rospy.ROSInterruptException:
        pass
