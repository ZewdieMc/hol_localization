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

def oplus(AxB,BxC):
    theta = AxB[2,0]
    AxC = np.array([
        [AxB[0,0] + BxC[0,0] * cos(theta) - BxC[1,0] * sin(theta)],
        [AxB[1,0] + BxC[0,0] * sin(theta) + BxC[1,0] * cos(theta)],
        [wrap_angle(theta + BxC[2,0])]])
    
    return AxC

def ominus(AxB):
        
    x = -AxB[0] * cos(AxB[2]) - AxB[1] * sin(AxB[2])
    y = AxB[0] * sin(AxB[2]) - AxB[1] * cos(AxB[2])
    yaw = wrap_angle(-AxB[2])
    return np.array([x, y, yaw]).reshape(-1,1)

def J1_oplus(xk_1,uk, dt):
    theta = wrap_angle(float(xk_1[2,0]))
    # Calculate Jacobians with respect to state vector#!(x, y, theta)
    J1_o = np.array([
                    [1, 0, -sin(theta) * uk[0,0]*dt - cos(theta) * uk[1,0]*dt],
                    [0, 1,  cos(theta) * uk[0,0]*dt - sin(theta) * uk[1,0]*dt],
                    [0, 0, 1                         ]
                    ])
    return J1_o

def J1_oplus_displacement(xk_1,uk):
    theta = wrap_angle(float(xk_1[2,0]))
    # Calculate Jacobians with respect to state vector#!(x, y, theta)
    J1_o = np.array([
                    [1, 0, -sin(theta) * uk[0,0] - cos(theta) * uk[1,0]],
                    [0, 1,  cos(theta) * uk[0,0] - sin(theta) * uk[1,0]],
                    [0, 0, 1                         ]
                    ])
    return J1_o

def J2_oplus(xk_1,dt):
    theta = wrap_angle(float(xk_1[2,0]))
    # Calculate Jacobians with respect to state vector#!(x, y, theta)
    J2_o = np.array([[cos(theta)*dt,    -sin(theta)*dt,   0.0],
                    [sin(theta)*dt,     cos(theta)*dt,   0.0],
                    [0.0,               0.0,         dt]])
    
    return J2_o

def J2_oplus_displacement(xk_1):
    theta = wrap_angle(float(xk_1[2,0]))
    # Calculate Jacobians with respect to state vector#!(x, y, theta)
    J2_o = np.array([[cos(theta),    -sin(theta),   0.0],
                    [sin(theta),     cos(theta),   0.0],
                    [0.0,               0.0,         1]])
    
    return J2_o

def J_ominus(AxB):
    J = np.array([[-cos(AxB[2,0]),    -sin(AxB[2,0]),   AxB[0,0]*sin(AxB[2,0])-AxB[1,0]*cos(AxB[2,0])],
                    [ sin(AxB[2,0]),    -cos(AxB[2,0]),   AxB[0,0]*cos(AxB[2,0])+AxB[1,0]*sin(AxB[2,0])],
                    [ 0.0,               0.0,            -1.0]])

    return J

def pose_prediction(xk_1,Pk_1,uk,Qk, Wk = 0,dt =1):
    # Calculate Jacobians with resp(J2_o @ self.A)ect to noise #!(vr, vl)
    J1_o = J1_oplus(xk_1,uk,dt)
    J2_o = J2_oplus(xk_1,dt)

    # pose update
    xk_bar = oplus(xk_1, uk*dt )
    Pk_bar = (J1_o @ Pk_1 @ J1_o.T) + (J2_o @ Qk @ J2_o.T ) + Wk


    return xk_bar, Pk_bar