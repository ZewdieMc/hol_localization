#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf
from utils.ekf_utils import pose_prediction, wrap_angle, J1_oplus
import scipy

class PEKFSLAM:
    def __init__(self, xk_0, Pk_0):   
        #robot pose and uncertainty
        self.xk = xk_0
        self.Pk = Pk_0
        self.dt = 0.01

        self.augment_threshold = 0.005


#######################################
####### Main Calculation
#######################################
    def prediction(self,uk,Qk):
        Wk = np.eye(3) * 0.000
        # Previous pose and covariance
        xk = self.xk[-3:,0].reshape(-1,1)
        Pk = self.Pk[-3:,-3:].reshape(3,3)
        xk_bar, Pk_bar = pose_prediction(xk,Pk,uk,Qk,Wk,self.dt)
        
        # Predict only current position
        self.xk[-3:,0] = xk_bar.reshape(3)

        # New Covariance 
        self.Pk[-3:,-3:] = Pk_bar

        J1_o = J1_oplus(xk, uk,self.dt)
    
        self.Pk[:-3, -3:] = self.Pk[:-3,-3:] @ J1_o.T
        self.Pk[-3:, :-3] = J1_o @ self.Pk[-3:,:-3] 

        
        # rospy.logerr("xk:{} ".format(self.xk))
        # rospy.logerr("Pk:{} ".format(self.Pk))
        return self.xk, self.Pk

    
    def update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk, h):
        kk = Pk_bar @ Hk.T @ np.linalg.inv(Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T)
        xk = xk_bar + kk @ wrap_angle(zk - h)
        I = np.eye(xk.shape[0])
        Pk = (I - kk @ Hk) @ Pk_bar

        xk[-1,0] = wrap_angle(xk[-1,0])

        return xk, Pk

#######################################
#######  Special Predict and Update
#######################################

    def update_imu(self, zk, Rk):
        h = np.array([wrap_angle(self.xk[-1,0])])
        Hk = np.zeros((zk.shape[0],self.xk.shape[0]))
        Hk[0,-1] = 1
        Vk = np.eye(zk.shape[0])
        xk,Pk = self.update(zk, Rk, self.xk, self.Pk, Hk, Vk, h)

        self.xk = xk
        self.Pk = Pk


#######################################
#######  State Augment Related
#######################################
    def augment_state(self):
        '''
        Augment current pose as a new state and also add covariance
        '''
        xk = self.xk
        Pk = self.Pk

        xk_plus = np.block([[xk],[xk[-3:,0].reshape(-1,1)]])

        Pk_plus = scipy.linalg.block_diag(Pk, Pk[-3:,-3:])
        if xk.shape[0] >3: # first covariance is only 3x3 so cannot use -3 index
            Pk_plus[:-3, -3:] = Pk[:,-3:]
            Pk_plus[-3:, :-3] = Pk[-3:,:]
        else:
            Pk_plus[:-3, -3:] = Pk
            Pk_plus[-3:, :-3] = Pk

        self.xk = xk_plus
        self.Pk = Pk_plus
        return xk_plus, Pk_plus


    def need_augment(self):
        '''
        To prevent unnecessary state augmentation
        '''
        if self.xk.shape[0] >3: 
            current_pose = self.xk[-3:,0].reshape(-1,1)
            lastest_vp = self.xk[-6:-3,0].reshape(-1,1)

            dif = current_pose - lastest_vp
            dif[0,-1] = wrap_angle(dif[0,-1])
            dif_norm = np.linalg.norm(dif)

            if dif_norm > self.augment_threshold:
                rospy.logwarn("need to augment")
                return True
            else:
                return False
        else:
            return True
             
        