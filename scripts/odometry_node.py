#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf

def wrap_angle(ang):
    return ang + (2.0 * np.pi * np.floor((np.pi - ang) / (2.0 * np.pi)))
class DeadReckoning:
    def __init__(self, joint_state_topic, odom_topic, Wb=0.235, Wr=0.035):

        
        # wheel joint names
        self.left_wheel_name = "turtlebot/kobuki/wheel_left_joint"
        self.right_wheel_name = "turtlebot/kobuki/wheel_right_joint"


        self.Qw = np.array([[np.deg2rad(10)**2, 0],
                            [0, np.deg2rad(10)**2]])
        
        #robot pose and uncertainty
        self.xk = np.array([0, 0, 0]).reshape(-1,1)
        self.Pk = np.array([[0.0001, 0 ,0],
                            [0, 0.00001, 0],
                            [0, 0 ,0.0001]])

        # wheel velocitiesWb
        self.left_vel = None
        self.right_vel = None
        self.dt = 0.01

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

        # odom measurement
        self.uk = np.array([0, 0, 0]).reshape(-1,1)

        # ODOM PUBLISHER
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=1)

        # JOINT STATE SUBSCRIBER #! Gives the wheel angular velocities
        self.joint_state_sub = rospy.Subscriber(joint_state_topic, JointState, self.joint_state_callback)


    def joint_state_callback(self, data):

        if data.name[0] == self.left_wheel_name:
            self.left_vel = data.velocity[0]
            self.left_vel_arrived = True
            rospy.logwarn("Left wheel: {}".format(self.left_vel) )

        if data.name[0] == self.right_wheel_name:
            self.right_vel = data.velocity[0]
            self.right_vel_arrived = True
            rospy.logwarn("Right wheel: {}".format(self.right_vel))
            
        if self.left_vel_arrived and self.right_vel_arrived:
            rospy.logwarn("Odom")
            #! Reset the flag
            self.left_vel_arrived = False
            self.right_vel_arrived = False

            xk = self.xk
            Pk = self.Pk

            uk, Qk = self.get_input()
            print(uk)
            xk_bar, Pk_bar =  self.prediction(xk,Pk , uk, Qk)

            # Update current state
            self.xk  = xk_bar
            self.Pk  = Pk_bar

            print(Pk_bar)
            
            # Publish the predicted odometry to rviz
            self.odom_path_pub()
            rospy.logwarn("Publish")
            rospy.logwarn("--------------------------------")


                
                

    def get_input(self):
        # Linear wheel velocities
       

        # delta t, time difference between two consecutive odometry readings
        self.dt = (rospy.Time.now() - self.last_time).to_sec()
        
        #update last time
        self.last_time = rospy.Time.now()

        # self.A = np.array([ #! not sure
        #     [0.5*self.Wr*self.dt,           0.5*self.Wr*self.dt],
        #     [0.0005,                             0.0005],
        #     [(0.5 * self.Wr*self.dt)/self.Wb, -(0.5 * self.Wr*self.dt)/self.Wb],
                
        #         ])
        dt = self.dt
        # self.A = np.array([ #! not sure
        #     [0.5*self.Wr*dt,           0.5*self.Wr*dt],
        #     [0.0,                             0.0],
        #     [(self.Wr*dt)/self.Wb, -(self.Wr*dt)/self.Wb],
                
        #         ])
        A =  np.array([[0.5 ,0.5],[0,0],[-1/self.Wb,1/self.Wb]]) @ np.diag([dt,dt]) @ np.diag([self.Wr,self.Wr])
        # r  = self.Wr
        # b = self.Wb
        # t = self.dt
        # dtheta = self.w * self.dt
        # d = self.v * self.dt
        # Aj = np.array([
        #     [r*t/2 * cos(dtheta) - sin(dtheta)*r*t/b * d, r*t/2 * cos(dtheta) + sin(dtheta)*r*t/b * d],
        #     [r*t/2 * sin(dtheta) + cos(dtheta)*r*t/b * d, r*t/2 * sin(dtheta) - cos(dtheta)*r*t/b * d],
        #     [r*t/b, -r*t/b]

        # ])
        
       # print(Aj)

        # self.Qk = np.array([[0.001, 0 ,0],
        #                     [0, 0.001, 0],
        #                     [0, 0 ,0.001]])

        #displacement
        # print(self.Qk)
        #displacement = np.array([self.v*self.dt, 0, self.w*self.dt])
        displacement = A @ np.array([[self.right_vel],[self.left_vel]])
        uk = displacement

        Qk =(A @ self.Qw @ A.T ) *10000
       # print(self.Qk)
       # self.Qk = np.diag([0.0001,0.00001,0.00001])
        #predicted state
        return uk, Qk

    def prediction(self,xk_1,Pk_1,uk,Qk):
        # local variables to make J2_o more readable
        
        theta = wrap_angle(float(xk_1[2,0]))
        # Calculate Jacobians with respect to state vector#!(x, y, theta)
        J1_o = np.array([
                      [1, 0, -sin(theta) * uk[0,0] - cos(theta) * uk[1,0]],
                      [0, 1,  cos(theta) * uk[0,0] - sin(theta) * uk[1,0]],
                      [0, 0, 1                         ]
                      ])
 
        # Calculate Jacobians with resp(J2_o @ self.A)ect to noise #!(vr, vl)
        J2_o = np.array([[cos(theta),    -sin(theta),   0.0],
                       [sin(theta),     cos(theta),   0.0],
                       [0.0,               0.0,         1]])
        
        # pose update
        xk_bar = self.oplus(xk_1, uk )
        # print(J1_o)
        # print(J2_o)
        # Prediction uncertainty
    
        Pk_bar = (J1_o @ Pk_1 @ J1_o.T) + (J2_o @ Qk @ J2_o.T )
        print(Pk_bar)
        return xk_bar, Pk_bar

    def oplus(self,AxB,BxC):
        theta = AxB[2,0]

        AxC = np.array([
            [AxB[0,0] + BxC[0,0] * cos(theta) - BxC[1,0] * sin(theta)],
            [AxB[1,0] + BxC[0,0] * sin(theta) + BxC[1,0] * cos(theta)],
            [wrap_angle(theta + BxC[2,0])]])
       
        return AxC

    def odom_path_pub(self):
        # Transform theta from euler to quaternion
        q = quaternion_from_euler(0, 0, float(wrap_angle(self.xk[2, 0])))

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

        odom.pose.covariance = list(np.array([[self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0, 2]],
                                [self.Pk[1, 0], self.Pk[1,1], 0, 0, 0, self.Pk[1, 2]],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2, 2]]]).flatten())

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w

        self.odom_pub.publish(odom)

        tf.TransformBroadcaster().sendTransform((float(self.xk[0, 0]), float(self.xk[1, 0]), 0.0), q, rospy.Time.now(
        ), odom.child_frame_id, odom.header.frame_id)


if __name__ == '__main__':

    rospy.init_node('Dead_reckoning')
    robot = DeadReckoning("/turtlebot/joint_states", "/odom")
    rospy.loginfo("Dead reckoning node started")

    rospy.spin()