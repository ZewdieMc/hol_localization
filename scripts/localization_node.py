#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf
from dead_reckoning import DeadReckoning
from scripts.PEKFSLAM import PEKFSLAM

def wrap_angle(ang):
    return ang + (2.0 * np.pi * np.floor((np.pi - ang) / (2.0 * np.pi)))
class LocalizationNode:
    def __init__(self, joint_state_topic, odom_topic, Wb=0.235, Wr=0.035):
        # Dead Reckoning Module
        self.dr = DeadReckoning(Wb, Wr)
        # wheel joint names
        self.left_wheel_name = "turtlebot/kobuki/wheel_left_joint"
        self.right_wheel_name = "turtlebot/kobuki/wheel_right_joint"
        # flags
        self.left_vel_arrived = False
        self.right_vel_arrived = False
        
        # Filter Module
        self.xk_0 = np.array([0, 0, 0]).reshape(-1,1)
        self.Pk_0 = np.array([[0.0001, 0 ,0],
                            [0, 0.00001, 0],
                            [0, 0 ,0.0001]])
        
        self.filter = PEKFSLAM(self.xk_0, self.Pk_0)
        
        self.robot_pose = self.get_robot_pose(self.xk_0, self.Pk_0)

        # Joint State Publisher
        self.joint_state_pub = rospy.Publisher(joint_state_topic, JointState, queue_size=1)

        # ODOM SUBSCRIBER
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        
        # ODOM PUBLISHER
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=1)

        # JOINT STATE SUBSCRIBER #! Gives the wheel angular velocities
        self.joint_state_sub = rospy.Subscriber(joint_state_topic, JointState, self.joint_state_callback)

        rospy.Timer(rospy.Duration(0.05), self.odom_pub)


    ##################################################################
    #### Callback functions
    ##################################################################
    def joint_state_callback(self, data):
        '''
        callback for /joint_state 
            - assign left and right velocity
            - calculate displacement using deadreckoning
            - do prediction
        '''
        if data.name[0] == self.left_wheel_name:
            self.left_vel = data.velocity[0]
            self.left_vel_arrived = True

        if data.name[0] == self.right_wheel_name:
            self.right_vel = data.velocity[0]
            self.right_vel_arrived = True
            
        if self.left_vel_arrived and self.right_vel_arrived:
            
            # set flags
            self.left_vel_arrived = False
            self.right_vel_arrived = False

            uk, Qk = self.dr.get_displacement(self.left_vel, self.right_vel)
            self.filter.prediction(uk, Qk)
         

    ##################################################################
    #### Timer Functions
    ##################################################################
    def odom_pub(self):
        '''
        Publish /odom and TF using current state vector
        '''
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"

        odom.pose = self.get_robot_pose(self.filter.xk, self.filter.Pk)

        self.odom_pub.publish(odom)

        q = quaternion_from_euler(0, 0, float(self.filter.xk[2, 0]))
        tf.TransformBroadcaster().sendTransform((float(self.filter.xk[0, 0]), float(self.filter.xk[1, 0]), 0.0), q, rospy.Time.now(
        ), odom.child_frame_id, odom.header.frame_id)


    ##################################################################
    #### Utility functions
    ##################################################################
    def get_robot_pose(self, xk, Pk):
        '''
        extract robot pose from state vector
        '''
        robot_pose.header.stamp = rospy.Time.now()
        robot_pose.header.frame_id = "world_ned"

        robot_pose = PoseWithCovarianceStamped()
        robot_pose.pose.pose.position.x = xk[0,0]
        robot_pose.pose.pose.position.y = xk[1,0]
        robot_pose.pose.pose.position.z = 0

        q = quaternion_from_euler(0, 0, float(xk[2, 0]))

        robot_pose.pose.pose.orientation.x = q[0]
        robot_pose.pose.pose.orientation.y = q[1]
        robot_pose.pose.pose.orientation.z = q[2]
        robot_pose.pose.pose.orientation.w = q[3]

        robot_pose.pose.covariance = list(np.array([[Pk[0, 0], Pk[0, 1], 0, 0, 0, Pk[0, 2]],
                                [Pk[1, 0], Pk[1,1], 0, 0, 0, Pk[1, 2]],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [Pk[2, 0], Pk[2, 1], 0, 0, 0, Pk[2, 2]]]).flatten())

        return robot_pose


if __name__ == '__main__':

    rospy.init_node('localization')
    robot = LocalizationNode("/turtlebot/joint_states", "/odom")
    rospy.loginfo("Localization node started")

    rospy.spin()