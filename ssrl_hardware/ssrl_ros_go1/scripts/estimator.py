#!/usr/bin/env python
from ssrl_ros_go1.estimator import Estimator
import rospy

if __name__ == '__main__':
    try:
        node = Estimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
