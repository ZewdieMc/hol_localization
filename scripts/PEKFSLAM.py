#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler
import tf
from utils.ekf_utils import pose_prediction

class PEKFSLAM:
    def __init__(self, xk_0, Pk_0):   
        #robot pose and uncertainty
        self.xk = xk_0
        self.Pk = Pk_0


    def prediction(self,uk,Qk):
        xk_bar, Pk_bar = pose_prediction(self.xk,self.Pk,uk,Qk)
        
        self.xk = xk_bar
        self.Pk = Pk_bar

    

   