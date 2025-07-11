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
    def __init__(self, joint_state_topic, odom_topic, Wb=0.235, Wr=0.035):

        # ODOM PUBLISHER
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=10)

        # JOINT STATE SUBSCRIBER #! Gives the wheel angular velocities
        self.joint_state_sub = rospy.Subscriber(joint_state_topic, JointState, self.joint_state_callback)

        # wheel joint names
        self.left_wheel_name = "turtlebot/kobuki/wheel_left_joint"
        self.right_wheel_name = "turtlebot/kobuki/wheel_right_joint"


        self.Qk = np.array([[0.2**2, 0, 0],
                            [0, 0.2**2, 0],
                            [0, 0, 0]])
        
        #robot pose and uncertainty
        self.xk = np.array([0, 0, 0]).reshape(-1,1)
        self.Pk = np.eye(3)*0.1

        # wheel velocitiesWb
        self.left_vel = None
        self.right_vel = None
        self.dt = 0

        # Wheel base
        self.Wb = Wb
        #wheel radius
        self.Wr = Wr

        # Linear and angular velocities
        self.v = 0
        self.w = 0

        self.last_time = rospy.Time.now()
       
        # both reading checker
        self.both_whls_rd = [False, False]

        # odom measurement
        self.uk = np.array([0, 0, 0]).reshape(-1,1)

    def joint_state_callback(self, data):
        if data.name[0] == self.left_wheel_name:
            self.left_vel = data.velocity[0]
            self.both_whls_rd[0] = True
        elif data.name[0] == self.right_wheel_name:
            self.right_vel = data.velocity[0]
            self.both_whls_rd[1] = True

            if all(self.both_whls_rd):
                self.dt = (rospy.Time.now() - self.last_time).to_sec()
                self.last_time = rospy.Time.now()
                self.compute_odometry()
                self.odom_path_pub()
                self.both_whls_rd = [False, False]

    def compute_odometry(self):
        # Linear wheel velocities
        vl = self.left_vel * self.Wr
        vr = self.right_vel * self.Wr

        # Linear and angular velocities of the robot
        self.v = (vr + vl) / 2
        self.w = (vl - vr)/ self.Wb

        #displacement
        self.uk = np.array([self.v * self.dt, 0, self.w * self.dt]).reshape(-1,1)

        #predicted state
        self.prediction()

    def prediction(self):
        # local variables to make J2_o more readable
        r = self.Wr
        dt = self.dt
        theta = float(self.xk[2,0])

        # Calculate Jacobians with respect to state vector#!(x, y, theta)
        J1_o = np.array([
                      [1, 0, -sin(theta) * self.uk[0, 0]],
                      [0, 1,  cos(theta) * self.uk[0, 0] ],
                      [0, 0, 1                                        ]
                      ])
 
        # Calculate Jacobians with respect to noise #!(vr, vl)
        # J2_o = np.array([
        #                 [0.5 * r * cos(theta) * dt,    0.5 * r * cos(theta) * dt],
        #                 [0.5 * r * sin(theta) * dt,    0.5 * r * sin(theta) * dt],
        #                 [(r / self.Wb) * dt,           -(r / self.Wb) * dt      ]
        #                 ])

        J2_o = np.array([
            [cos(theta), -sin(theta), 0.0],
            [sin(theta),  cos(theta), 0.0],
            [0.0,             0.0,            1.0]
        ])
        
        # pose update
        self.xk = self.oplus()

        # Prediction uncertainty
        self.Pk = J1_o @ self.Pk @ J1_o.T + J2_o @ self.Qk @ J2_o.T

    def oplus(self):
        AxB = self.xk
        BxC = self.uk
        theta = float(self.xk[2,0])

        AxC = np.array([
            AxB[0,0] + BxC[0,0] * cos(theta) - BxC[1,0] * sin(theta),
            AxB[1,0] + BxC[0,0] * sin(theta) + BxC[1,0] * cos(theta),
            wrap_angle(theta + BxC[2,0])
        ])
        return AxC.reshape(-1,1)

    def odom_path_pub(self):
        # Transform theta from euler to quaternion
        q = quaternion_from_euler(0, 0, float(wrap_angle(self.xk[2])))
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

        odom.pose.covariance = [self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0, 2],
                                self.Pk[1, 0], self.Pk[1,1], 0, 0, 0, self.Pk[1, 2],
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2, 2]]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w

        self.odom_pub.publish(odom)

        tf.TransformBroadcaster().sendTransform((float(self.xk[0, 0]), float(self.xk[1, 0]), 0.0), q, rospy.Time.now(
        ), odom.child_frame_id, odom.header.frame_id)


if __name__ == '__main__':

    rospy.init_node('Dead_reckoning')
    robot = DeadReckoning("/turtlebot/joint_states", "/turtlebot/odom")
    rospy.loginfo("Dead reckoning node started")

    rospy.spin()