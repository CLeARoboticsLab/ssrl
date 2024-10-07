#!/usr/bin/env python
from ssrl_ros_go1_msgs.msg import QuadrupedCommand

import rospy


def publish_command():
    rospy.init_node('command_pub')
    rospy.loginfo("Starting command publisher node")

    pub = rospy.Publisher('quadruped_command', QuadrupedCommand, queue_size=100)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        cmd_msg = QuadrupedCommand()
        cmd_msg.forward_vel = 0.0
        cmd_msg.turn_rate = 0.0

        pub.publish(cmd_msg)

        rate.sleep()


if __name__ == '__main__':
    try:
        publish_command()
    except rospy.ROSInterruptException:
        pass
