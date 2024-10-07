#include <ros/ros.h>
#include <ssrl_ros_go1_msgs/PdTarget.h>
#include <ssrl_ros_go1_msgs/TorqueTarget.h>
#include <unitree_legged_msgs/LowCmd.h>
#include "unitree_legged_sdk/unitree_legged_sdk.h"

using namespace UNITREE_LEGGED_SDK;

class LowCmdPub
{
  public:
    LowCmdPub()
    {
      const auto queue_size = 1000;
      pd_cmd_sub_ = nh_.subscribe("pd_target", queue_size, &LowCmdPub::pdCallback, this, ros::TransportHints().tcpNoDelay(true));
      tq_cmd_sub_ = nh_.subscribe("torque_target", queue_size, &LowCmdPub::torqueCallback, this, ros::TransportHints().tcpNoDelay(true));
      cmd_pub_ = nh_.advertise<unitree_legged_msgs::LowCmd>("low_cmd", queue_size);
      
      lcmd_.head[0] = 0xFE;
      lcmd_.head[1] = 0xEF;
      lcmd_.levelFlag = LOWLEVEL;
      for (int i = 0; i < 12; i++)
      {
        lcmd_.motorCmd[i].mode = 0x00;
        lcmd_.motorCmd[i].q = PosStopF;
        lcmd_.motorCmd[i].Kp = 0;
        lcmd_.motorCmd[i].dq = VelStopF;
        lcmd_.motorCmd[i].Kd = 0;
        lcmd_.motorCmd[i].tau = 0;
      }
    }

  private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub_;
    ros::Subscriber pd_cmd_sub_;
    ros::Subscriber tq_cmd_sub_;
    unitree_legged_msgs::LowCmd lcmd_;

    void pdCallback(ssrl_ros_go1_msgs::PdTarget pd)
    {
      for (int i = 0; i < 12; i++)
      {
        lcmd_.motorCmd[i].mode = pd.mode;
        lcmd_.motorCmd[i].q = pd.q_des[i];
        lcmd_.motorCmd[i].Kp = pd.Kp[i];
        lcmd_.motorCmd[i].dq = pd.qd_des[i];
        lcmd_.motorCmd[i].Kd = pd.Kd[i];
        lcmd_.motorCmd[i].tau = 0;
      }
      cmd_pub_.publish(lcmd_);
    }

    void torqueCallback(ssrl_ros_go1_msgs::TorqueTarget tq)
    {
      for (int i = 0; i < 12; i++)
      {
        lcmd_.motorCmd[i].mode = tq.mode;
        lcmd_.motorCmd[i].q = PosStopF;
        lcmd_.motorCmd[i].Kp = 0;
        lcmd_.motorCmd[i].dq = VelStopF;
        lcmd_.motorCmd[i].Kd = 0;
        lcmd_.motorCmd[i].tau = tq.tau_des[i];
      }
      cmd_pub_.publish(lcmd_);
    }
};

auto main(int argc, char **argv) -> int
{
  ros::init(argc, argv, "low_cmd_pub");
  LowCmdPub lcp;
  ROS_INFO_STREAM("Started low_cmd publisher" );
  ros::spin();
  return 0;
}