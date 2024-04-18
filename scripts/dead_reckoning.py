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
        self.dt = 0.01

        # Wheel base
        self.Wb = Wb
        #wheel radius
        self.Wr = Wr

        # noise in encoder's velocity
        self.Qw = np.array([[np.deg2rad(10)**2, 0],
                            [0, np.deg2rad(10)**2]])
        

        self.last_time = rospy.Time.now()

        # odom measurement
        self.uk = np.array([0, 0, 0]).reshape(-1,1)

    def get_displacement(self, left_vel, right_vel):
        # delta t, time difference between two consecutive odometry readings
        self.dt = (rospy.Time.now() - self.last_time).to_sec()
        
        #update last time
        self.last_time = rospy.Time.now()

        dt = self.dt
        A =  np.array([[0.5 ,0.5],[0,0],[-1/self.Wb,1/self.Wb]]) @ np.diag([dt,dt]) @ np.diag([self.Wr,self.Wr])
        
        displacement = A @ np.array([[right_vel],[left_vel]])
        uk = displacement

        Qk =(A @ self.Qw @ A.T ) 
   
        return uk, Qk

    