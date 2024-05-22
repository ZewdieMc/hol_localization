#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf
'''
Class to calculate displacement from encoder velocity
'''
def wrap_angle(ang):
    return ang + (2.0 * np.pi * np.floor((np.pi - ang) / (2.0 * np.pi)))
class DeadReckoning:
    def __init__(self, Wb=0.235, Wr=0.035):

        # wheel velocitiesWb
        self.dt = 0.001

        # Wheel base
        self.Wb = Wb
        #wheel radius
        self.Wr = Wr

        # noise in encoder's velocity
        self.Qw = np.array([[20**2, 0],
                            [0, 20**2]])
        

        self.last_time = rospy.Time.now()

        # odom measurement
        self.uk = np.array([0, 0, 0]).reshape(-1,1)

    def get_input(self, left_vel, right_vel):
        # delta t, time difference between two consecutive odometry readings
        dt = (rospy.Time.now() - self.last_time).to_sec()
        
        #update last time
        self.last_time = rospy.Time.now()

        A =  np.array([[0.5*self.Wr,      0.5*self.Wr],
                       [0,                 0             ],
                       [self.Wr/self.Wb, -self.Wr/self.Wb]])
        
        velocity = A @ np.array([[left_vel],[right_vel]])
        uk = velocity

        Qk =(A @ self.Qw @ A.T ) 
   
        return uk, Qk, dt

    