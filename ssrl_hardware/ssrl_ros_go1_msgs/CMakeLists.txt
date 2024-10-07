cmake_minimum_required(VERSION 2.8.3)
project(ssrl_ros_go1_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_message_files(
  FILES
  Observation.msg
  PdTarget.msg
  TorqueTarget.msg
  Gait.msg
  Estimation.msg
  QuadrupedState.msg
  QuadrupedCommand.msg
  Action.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
)

#############
## Install ##
#############

# Mark topic names header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)