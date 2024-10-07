cmake_minimum_required(VERSION 3.0.2)
project(ssrl_ros_go1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  ssrl_ros_go1_msgs
  # unitree_legged_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS roscpp rospy
)

message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
else()
  set(ARCH arm64)
endif()

link_directories(${CMAKE_SOURCE_DIR}/unitree_ros_to_real/unitree_legged_sdk/lib/cpp/${ARCH})

set(EXTRA_LIBS -pthread libunitree_legged_sdk.so)

set(CMAKE_CXX_FLAGS "-O3 -fPIC")

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/unitree_ros_to_real/unitree_legged_sdk/include
)

add_executable(low_cmd_pub
	src/low_cmd_pub.cpp
)
target_link_libraries(
	low_cmd_pub
	${catkin_LIBRARIES}
)

install(TARGETS	low_cmd_pub
  	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)