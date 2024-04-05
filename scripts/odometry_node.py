#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf


def wrap_angle(ang):
    if isinstance(ang, np.ndarray):
        ang[ang > np.pi] -= 2.0 * np.pi
        return ang
    else:
        return ang + (2.0 * np.pi * np.floor((np.pi - ang) / (2.0 * np.pi)))
class DeadReckoning:
    def __init__(self, joint_state_topic, odom_topic, Wb=0.23, Wr=0.035):

        # ODOM PUBLISHER
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=10)

        # JOINT STATE SUBSCRIBER
        self.joint_state_sub = rospy.Subscriber(joint_state_topic, JointState, self.joint_state_callback)

        # wheel joint names
        self.left_wheel_name = "turtlebot/kobuki/wheel_left_joint"
        self.right_wheel_name = "turtlebot/kobuki/wheel_right_joint"

        #robot pose and uncertainty
        self.xk = np.array([[0], [0], [0]])
        self.PK = np.eye(3)*0.1

        # wheel velocitiesWb
        self.left_vel = None
        self.right_vel = None
        self.dt = None

        # Wheel base
        self.Wb = Wb
        #wheel radius
        self.Wr = Wr

        # Linear and angular velocities
        self.v = 0
        self.w = 0

        self.last_time = rospy.Time.now()
        self.left_vel_arrived = False
        self.right_vel_arrived = False


    def joint_state_callback(self, data):
        print("Joint state callback")
        print("data: name: ", data.name[0])
        if data.name[0] == self.left_wheel_name:
            print("Left wheel velocity arrived")
            self.left_vel = data.velocity[0]
            self.left_vel_arrived = True
        elif data.name[0] == self.right_wheel_name:
            print("Right wheel velocity arrived")
            self.right_vel = data.velocity[0]
            self.right_vel_arrived = True
            
        if self.left_vel_arrived and self.right_vel_arrived:
            self.compute_odometry()

            # Publish the predicted odometry to rviz
            self.odom_path_pub()

            #! Reset the flags
            self.left_vel_arrived = False
            self.right_vel_arrived = False

    def compute_odometry(self):
        # Linear wheel velocities
        vl = self.left_vel * self.Wr
        vr = self.right_vel * self.Wr

        self.v = (vr + vl) / 2
        self.w = (vl - vr)/ self.Wb

        # delta t, time difference between two consecutive odometry readings
        self.dt = (rospy.Time.now() - self.last_time).to_sec()
        
        #update last time
        self.last_time = rospy.Time.now() #! Note sure

        # Compute the new pose(prediction)
        #displacement
        displacement = np.array([self.v * self.dt, 0, self.w * self.dt])

        #predicted state
        self.xk = self.oplus(self.xk, displacement)

    def prediction(self):
        # Jacobian wrt the state
        Ak = np.array([
                    [1.0, 0.0, ...],
                    [0.0, 1.0, ...],
                    [0.0, 0.0, 1.0],
        ])

        # Jacobian wrt noise
        Bk = np.array([
                    [],
                    [],
                    []
        ])


        #displacement
        displacement = np.array([self.v * self.dt, 0, self.w * self.dt])

        #predicted state
        xk = self.oplus(self.xk, displacement)

        #predicted covariance
        PK = None

        return xk, PK


    def oplus(self, AxB, BxC):
        x = AxB[0] + BxC[0] * cos(AxB[2]) - BxC[1] * sin(AxB[2])
        y = AxB[1] + BxC[0] * sin(AxB[2]) + BxC[1] * cos(AxB[2])
        yaw = AxB[2] + BxC[2]
        return np.array([x, y, yaw])

    def odom_path_pub(self):
        print("Publishing odom")
        # Transform theta from euler to quaternion
        q = quaternion_from_euler(0, 0, float(self.xk[2]))

        # Publish predicted odom
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"

        odom.pose.pose.position.x = self.xk[0]
        odom.pose.pose.position.y = self.xk[1]

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # odom.pose.covariance = [self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0, 2],
        #                         self.Pk[1, 0], self.Pk[1,1], 0, 0, 0, self.Pk[1, 2],
        #                         0, 0, 0, 0, 0, 0,
        #                         0, 0, 0, 0, 0, 0,
        #                         0, 0, 0, 0, 0, 0,
        #                         self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2, 2]]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w

        self.odom_pub.publish(odom)

        tf.TransformBroadcaster().sendTransform((float(self.xk[0]), float(self.xk[1]), 0.0), q, rospy.Time.now(
        ), odom.child_frame_id, odom.header.frame_id)


if __name__ == '__main__':

    rospy.init_node('Dead_reckoning')
    robot = DeadReckoning("/turtlebot/joint_states", "/odom")
    rospy.loginfo("Dead reckoning node started")

    rospy.spin()