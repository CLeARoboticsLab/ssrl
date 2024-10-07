# ssrl_hardware

ROS is used to interface with the Go1 robot and perform real-world experiments. Below are instructions for running experiments with the Go1.

## Installation

Copy the contents of this package to your `catkin_ws/src`. Use `catkin make` to build the package.

## Setup

+ Vicon
    1. Calibrate the Vicon system per normal procedures.
    2. Create a tracker object in the Vicon software for the Go1 named `quad`. Ensure that the x-axis of the tracker is aligned with the front of the Go1. (Do this by aligning the Go1 with the global x-axis and then creating the tracker object.)

+ Go1
    1. Ensure the Go1 is laying prone on the ground and aligned with the global x-axis.
    2. Power on the Go1 and the remote.
    3. After the Go1 has booted up and is standing up, input the following on the remote to make the Go1 go prone and accept low level commands:
       + L2 + A
       + L2 + A
       + L2 + B
       + L1 + L2 + Start
    4. Hang up the Go1 and connect the ethernet cable to the laptop.

+ Laptop
    1. Make sure the laptop is connected to the network as the Vicon.
    2. Make sure the laptop's ethernet IP address is set to `192.168.123.XXX`
    3. In 3 separate terminal tabs/windows, run the following commands:
       + `roscore`
       + `roslaunch ssrl_ros_go1 quadruped_comm.launch`
       + `roslaunch ssrl_ros_go1 support_nodes_vicon.launch`

## Running Experiments

1. Start the controller by running the following command in a new terminal tab/window:
    + `roslaunch ssrl_ros_go1 controller.launch`
2. Generate data:
    + Press the spacebar to start standing up the Go1.
    + When the Go1 is standing, press the spacebar again to start walking.
    + The episode will terminiate automatically after 10 sec or if the robot falls.
    + To stop the episode early, press any key.
3. Train:
    + Run the following to run training: `python ssrl_hardware/ssrl_ros_go1/scripts/train.py run_name=<RUN_NAME>`
        + Replace `<RUN_NAME>` with a descriptive name of the run. The name must not contain spaces or special characters except underscore.
4. Repeat for the desired number of epochs.
