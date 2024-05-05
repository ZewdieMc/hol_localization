#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf
from utils.ekf_utils import pose_prediction, wrap_angle, J1_oplus, oplus, ominus, J1_oplus_displacement, J2_oplus_displacement, J_ominus
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

    
    def update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk, h,zk_h):
        kk = Pk_bar @ Hk.T @ np.linalg.inv(Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T)
        xk = xk_bar + kk @ zk_h
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
        zk_h =  wrap_angle(zk-h)
        xk,Pk = self.update(zk, Rk, self.xk, self.Pk, Hk, Vk, h, zk_h)

        self.xk = xk
        self.Pk = Pk

    def update_lm(self, zlm, Rlm, hlm, hypothesis):
        rospy.logwarn("Update Laser Matching!!!!")
        Hk = np.zeros((zlm.shape[0], self.xk.shape[0]))
        Vk = np.eye(zlm.shape[0])

        current_state = self.xk[-3:,0].reshape(-1,1)
        
        J2_o = J2_oplus_displacement(current_state)
        J_omi = J_ominus(current_state)
        print(hypothesis)
        for i,hypo in enumerate(hypothesis):
            hypo_state = self.xk[hypo*3:hypo*3+3,0].reshape(-1,1)
            J1_o = J1_oplus_displacement(ominus(current_state),hypo_state )
            
            Hk[i*3:i*3+3, hypo*3:hypo*3+3] = J2_o
            Hk[i*3:i*3+3, -3:] = J1_o @ J_omi
         

        zk_h = zlm - hlm
        for i in range(zk_h.shape[0]):
            if i+1 % 3 == 0:
                zk_h[i,0] = wrap_angle(zk_h[i])
        
        rospy.logerr("zlm: {}".format(zlm))
        rospy.logerr("hlm: {}".format(hlm))
        rospy.logerr("zk_h: {}".format(zk_h))


        xk,Pk = self.update(zlm, Rlm, self.xk, self.Pk, Hk, Vk, hlm, zk_h)

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

            dif = current_pose[:2] - lastest_vp[:2]
            # dif[0,-1] = wrap_angle(dif[0,-1])
            dif_norm = np.linalg.norm(dif)

            if dif_norm > self.augment_threshold:
                rospy.logwarn("need to augment")
                return True
            else:
                return False
        else:
            return True
        
    def find_initial_guess(self, previous_pose, current_pose, previous_pose_cov, current_pose_cov):
        dis = oplus(ominus(previous_pose), current_pose)
        # claculate displacement
        dx = dis[0,0]
        dy = dis[1,0]
        dyaw = dis[2,0]
        q = quaternion_from_euler(0, 0, float(dyaw))

        initial_guess = np.array([dx,dy,dyaw]).reshape(-1,1)
        J1_o = J1_oplus_displacement(ominus(previous_pose),current_pose)
        J2_o = J2_oplus_displacement(ominus(previous_pose))
        J_omi = J_ominus(previous_pose)
        initial_guess_cov = (J1_o @ J_omi @ previous_pose_cov @ J_omi.T @ J1_o.T) + (J2_o @ current_pose_cov @ J2_o.T) 
        # service response
        initial_guess_transform = Transform()
        initial_guess_transform.translation.x = dx
        initial_guess_transform.translation.y = dy
        initial_guess_transform.translation.z = 0
        initial_guess_transform.rotation.x = q[0]
        initial_guess_transform.rotation.y = q[1]
        initial_guess_transform.rotation.z = q[2]
        initial_guess_transform.rotation.w = q[3]

        return initial_guess_transform, initial_guess, initial_guess_cov
    
    def individual_compatable(self,scan_matching,initial_guess,scan_matching_cov,initial_guess_cov):
        return True
        