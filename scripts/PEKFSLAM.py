#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf
from utils.ekf_utils import pose_prediction, wrap_angle

class PEKFSLAM:
    def __init__(self, xk_0, Pk_0):   
        #robot pose and uncertainty
        self.xk = xk_0
        self.Pk = Pk_0


    def prediction(self,uk,Qk):
        Wk = np.eye(3) * 0.0001
        xk_bar, Pk_bar = pose_prediction(self.xk,self.Pk,uk,Qk,Wk)
        
        self.xk = xk_bar
        self.Pk = Pk_bar
        return xk_bar, Pk_bar

    def update_imu(self, zk, Rk):
        h = self.xk[2,0]
        Hk = np.zeros((zk.shape[0],self.xk.shape[0]))
        print(Hk)
        Hk[0,2] = 1
        print(Hk)
        Vk = np.eye(zk.shape[0])
        xk,Pk = self.update(zk, Rk, self.xk, self.Pk, Hk, Vk, h)

        self.xk = xk
        self.Pk = Pk

    def update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk, h):
        
        kk = Pk_bar @ Hk.T @ np.linalg.inv(Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T)
        xk = xk_bar + kk @ (zk - h)
        I = np.eye(xk.shape[0])
        Pk = (I - kk @ Hk) @ Pk_bar

        xk[2,0] = wrap_angle(xk[2,0])

        return xk, Pk
    

   