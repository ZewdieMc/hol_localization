# subscribe to the encoder and #!IMU data, and publish odometry data
import rospy
from sensor_msgs.msg import JointState
from  nav_msgs.msg import Odometry
import numpy as np
from numpy import cos, sin

class DeadReckoning:
    def __init__(self, joint_state_topic, odom_topic):

        self.odom_pub = rospy.Publisher('odometry', Odometry, queue_size=10)
        self.joint_state_sub = rospy.Subscriber(joint_state_topic, JointState, self.joint_state_callback)

        # class variables
        self.left_wheel_name = "turtlebot/kobuki/wheel_encoder_left"
        self.right_wheel_name = "turtlebot/kobuki/wheel_encoder_right"

        self.xk = np.array([[0], [0], [0]])
        self.left_vel = 0
        self.right_vel = 0
        self.dt = None
        self.last_time = rospy.Time.now()
        self.vel_arrived = False

        # deadreckoning parameters
        self.wheel_base = 0.287

        self.odom = Odometry()
        self.odom.header.frame_id = 'odom'
        self.odom.child_frame_id = 'base_link'
        

    def joint_state_callback(self, data):
        if data.name == 'left_wheel_joint':
            self.left_vel = data.velocity[0]

            
            self.compute_odometry()

    def compute_odometry(self):
        vl = self.left_vel
        vr = self.right_vel

        vx = (vr + vl) / 2
        w = (vr - vl)/ self.wheel_base

        dx = vx * self.dt
        zeta = w * self.dt

        displacement = np.array([[dx], [0], [zeta]])
        
        self.xk = self.oplus(self.xk,displacement)

    def oplus(self, AxB, BxC):
                # TODO: to be completed by the student
        x = AxB[0] + BxC[0] * cos(AxB[2]) - BxC[1] * sin(AxB[2])
        y = AxB[1] + BxC[0] * sin(AxB[2]) + BxC[1] * cos(AxB[2])
        yaw = AxB[2] + BxC[2]
        return np.array([x, y, yaw])

    def imu_callback(self, data):
        pass