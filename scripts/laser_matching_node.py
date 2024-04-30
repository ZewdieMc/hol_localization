#!/usr/bin/python3

import rospy
from  nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Transform
import numpy as np
from numpy import cos, sin
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from ho_localization.srv import OdomTransform, PointCloudTransform, PointCloudTransformRequest
from utils.ekf_utils import oplus
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from std_srvs.srv import Trigger, TriggerResponse
class LaserMatchingNode:
    def __init__(self, pc_topic, lm_odom_topic):
        
        # Tuning parameters
        self.matching_interval = 2
        self.publish_tf = False
        self.moving_threshold = 0.001
        self.rotation_threshold = np.pi / 200


        # Class variables
        self.scan_frame = "turtlebot/kobuki/rplidar"
        self.bf_frame = "turtlebot/kobuki/base_footprint"
        self.pc_list = []
        self.initial_guese = TransformStamped()
        self.current_pc = None
        # For Plotting
        self.target_pc_list = [] 
        self.current_pc_list = []
        self.initial_guese_list = []
        self.matching_tf_list = []
        self.process_time_list = []
        self.gt_list = []


        # Initial state
        self.xk = np.array([3.0, -0.78, np.pi/2]).reshape(-1,1)
        # self.Pk_0 = np.array([[0.0001, 0 ,0],
        #                     [0, 0.00001, 0],
        #                     [0, 0 ,0.0001]])
        
        # TF Listener 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # LM ODOM PUBLISHER
        self.lm_odom_pub = rospy.Publisher(lm_odom_topic, Odometry, queue_size=1)
        self.pc_pub = rospy.Publisher("/pc_tf", PointCloud2, queue_size=1)

        # POINT CLOUD SUBSCRIBER
        self.pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_cb)

        # Intialize first pc 
        while not self.current_pc:
            rospy.logwarn("Waiting for first point cloud...")
            rospy.sleep(1)
        self.pc_list.append(self.current_pc)

        rospy.loginfo("Fininsh Initilize -- Start matching loop...")

        # Main exeution loop (need to wait all the initial data to arrive)
        rospy.Timer(rospy.Duration(self.matching_interval), self.matching_loop)

        # For odom publishing
        rospy.Timer(rospy.Duration(0.05), self.odom_msg_pub)

        self.plotting_srv = rospy.Service('plot_matching', Trigger, self.handle_plot_matching)



    ##################################################################
    #### Callback functions
    ##################################################################
    def pc_cb(self, data):
        '''
        callback for point cloud 
            
        '''
        # Transform point cloud to reference with base_footprint
        pc = self.pc_in_footprint(data)

        self.current_pc = pc 
        self.pc_pub.publish(self.current_pc)

    def handle_plot_matching(self,req):
        rospy.loginfo("handle_plot_matching")
        self.plot_matching(self.target_pc_list, self.current_pc_list, self.matching_tf_list, self.initial_guese_list, self.process_time_list, self.gt_list)

        rospy.loginfo("Finish plotting")
        self.target_pc_list = [] 
        self.current_pc_list = []
        self.initial_guese_list = []
        self.matching_tf_list = []
        self.process_time_list = []
        self.gt_list = []

        res = TriggerResponse()
        return res
    
        
    ##################################################################
    #### Core functions
    ##################################################################
    def pc_in_footprint(self, pc_msg):
        
        target_frame = self.bf_frame  # Replace with your target frame
        source_frame = self.scan_frame  # Replace with your source frame

        # Wait for the transform to be available
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
        pc_transformed = do_transform_cloud(pc_msg, transform)
        return pc_transformed

    def get_initial_guess(self):
        rospy.logwarn("Calling get_initial_guess...")
        rospy.wait_for_service('get_initial_guess')
        try:
            get_inital_guess = rospy.ServiceProxy('get_initial_guess', OdomTransform)
            resp = get_inital_guess()
            rospy.logwarn("initial guess:{}".format(resp.transform))
            return resp.transform
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return None
        
    def get_gt(self):
        rospy.logwarn("Calling get_gt...")
        rospy.wait_for_service('get_gt')
        try:
            get_inital_guess = rospy.ServiceProxy('get_gt', OdomTransform)
            resp = get_inital_guess()
            rospy.logwarn("gt:{}".format(resp.transform))
            return resp.transform
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return None
        
    def get_matching_tf(self,initial_guess,target_pc,current_pc):
        rospy.logwarn("Calling get_matching_tf...")
        rospy.wait_for_service('ndt_matching')
        try:
            get_matching_tf = rospy.ServiceProxy('ndt_matching', PointCloudTransform)
            req = PointCloudTransformRequest()
            req.initial_guese = initial_guess
            req.target = target_pc
            req.current = current_pc
            resp = get_matching_tf(req)
            rospy.logwarn("matching: {}".format(resp.transform))

            return resp.transform
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return None


    def update_pose(self,xk, tf):
        '''
        update state vector
        '''
        dx = tf.translation.x
        dy = tf.translation.y
        q = (tf.rotation.x, tf.rotation.y,tf.rotation.z,tf.rotation.w)
        _,_,dyaw = euler_from_quaternion(q)

        displacement = np.array([[dx],[dy],[dyaw]])
        xk = oplus(xk, displacement)
        return xk
    
    def robot_moving(self,initial_guess):
        x = initial_guess.translation.x
        y = initial_guess.translation.y
        q = (initial_guess.rotation.x, initial_guess.rotation.y,initial_guess.rotation.z,initial_guess.rotation.w)        
        _,_,yaw = euler_from_quaternion(q)

        if abs(x) < self.moving_threshold and abs(y) < self.moving_threshold and abs(yaw) < self.rotation_threshold:
            return False
        else:
            return True
    ##################################################################
    #### Timer Functions
    ##################################################################
    def matching_loop(self,_):
        '''
        
        '''
        
        # update pc_list
        self.pc_list.append(self.current_pc)

        # request initial guese from odom node
        initial_guese =  self.get_initial_guess()
        gt = self.get_gt()

        # request mathcing tf
        if self.robot_moving(initial_guese):
            target_pc = self.pc_list[-2]
            current_pc = self.pc_list[-1]
            
            start = rospy.Time.now()
            matching_tf = self.get_matching_tf(initial_guese,target_pc,current_pc)
            process_time = (rospy.Time.now()-start).to_sec()
            rospy.logerr("Time used: {}".format(process_time))

            self.target_pc_list.append(target_pc)
            self.current_pc_list.append(current_pc)
            self.initial_guese_list.append(initial_guese)
            self.matching_tf_list.append(matching_tf)
            self.process_time_list.append(process_time)
            self.gt_list.append(gt)
        else:
            matching_tf = Transform()
        # update predited state
        self.xk = self.update_pose(self.xk, matching_tf)

        
        
        


    def odom_msg_pub(self,_):
        '''
        Publish /odom and TF using current state vector
        '''
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = self.bf_frame

        odom.pose = self.get_robot_pose(self.xk, np.zeros((3,3))).pose

        self.lm_odom_pub.publish(odom)

        q = quaternion_from_euler(0, 0, float(self.xk[2, 0]))
        if self.publish_tf:
            tf.TransformBroadcaster().sendTransform((float(self.xk[0, 0]), float(self.xk[1, 0]), 0.0), q, rospy.Time.now(
            ), odom.child_frame_id, odom.header.frame_id)
        

    ##################################################################
    #### Utility functions
    ##################################################################
    def get_robot_pose(self, xk, Pk):
        '''
        extract robot pose from state vector
        '''
        robot_pose = PoseWithCovarianceStamped()
        robot_pose.header.stamp = rospy.Time.now()
        robot_pose.header.frame_id = "world_ned"

        robot_pose.pose.pose.position.x = xk[0,0]
        robot_pose.pose.pose.position.y = xk[1,0]
        robot_pose.pose.pose.position.z = 0

        q = quaternion_from_euler(0, 0, float(xk[2, 0]))

        robot_pose.pose.pose.orientation.x = q[0]
        robot_pose.pose.pose.orientation.y = q[1]
        robot_pose.pose.pose.orientation.z = q[2]
        robot_pose.pose.pose.orientation.w = q[3]

        robot_pose.pose.covariance = list(np.array([[Pk[0, 0], Pk[0, 1], 0, 0, 0, Pk[0, 2]],
                                [Pk[1, 0], Pk[1,1], 0, 0, 0, Pk[1, 2]],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [Pk[2, 0], Pk[2, 1], 0, 0, 0, Pk[2, 2]]]).flatten())

        return robot_pose
    
    def plot_matching(self,target_pc,current_pc,matching_tf, initial_guess, process_time, gt):
        # Check if both point clouds are available
        for i in range(len(target_pc)):
            # Extract point cloud data from messages
            target_points = pc2.read_points(target_pc[i], field_names=("x", "y"), skip_nans=True)
            target_points1 = copy(list(target_points))
            target_points2 = copy(target_points1)
            target_points3 = copy(target_points1)
            target_points4 = copy(target_points1)
            current_points = pc2.read_points(current_pc[i], field_names=("x", "y"), skip_nans=True)

            # Plot before matching (current_pc)
            fig = plt.figure(figsize=(20, 20))
            ax1 = fig.add_subplot(221)
            for point in current_points:
                ax1.scatter(point[0], point[1], c='b', marker='.',s=0.3)
            for point in target_points1:
                ax1.scatter(point[0], point[1], c='r', marker='x',s=1)
            
            

            ax1.set_title('Before Matching ')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            
            
            # Transform pc using tf
            matching_tf_stamped = TransformStamped()
            matching_tf_stamped.transform = matching_tf[i]
            pc_aligned = do_transform_cloud(current_pc[i], matching_tf_stamped)
            pc_aligned_points = pc2.read_points(pc_aligned, field_names=("x", "y"), skip_nans=True)

            # Plot after matching (target_pc transformed by matching_tf)
            ax2 = fig.add_subplot(222)
            for point in pc_aligned_points:
                ax2.scatter(point[0], point[1], c='b', marker='.',s=0.3)
            for point in target_points2:
                ax2.scatter(point[0], point[1], c='r', marker='x',s=0.3)
            
            x = round(matching_tf[i].translation.x,4)
            y = round(matching_tf[i].translation.y,4)
            q = (matching_tf[i].rotation.x, matching_tf[i].rotation.y,matching_tf[i].rotation.z,matching_tf[i].rotation.w)
            _,_,yaw = euler_from_quaternion(q)
            yaw = round(yaw,4)


            ax2.set_title('After Matching \n x:{} \n y:{} \n yaw:{} \n ({} s)'.format(x,y,yaw,round(process_time[i],2)))
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')

            # Transform pc using initial guese
            matching_tf_stamped = TransformStamped()
            matching_tf_stamped.transform = initial_guess[i]
            pc_odom = do_transform_cloud(current_pc[i], matching_tf_stamped)
            pc_odom_points = pc2.read_points(pc_odom, field_names=("x", "y"), skip_nans=True)

            # Plot after matching (target_pc transformed by odom)
            ax3 = fig.add_subplot(223)
            for point in pc_odom_points:
                ax3.scatter(point[0], point[1], c='b', marker='.',s=0.3)
            for point in target_points3:
                ax3.scatter(point[0], point[1], c='r', marker='x',s=0.3)

            x = round(initial_guess[i].translation.x,4)
            y = round(initial_guess[i].translation.y,4)
            q = (initial_guess[i].rotation.x, initial_guess[i].rotation.y,initial_guess[i].rotation.z,initial_guess[i].rotation.w)
            _,_,yaw = euler_from_quaternion(q)
            yaw = round(yaw,4)

            ax3.set_title('After Matching (Odom) \n x:{} \n y:{} \n yaw:{}'.format(x,y,yaw))
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')

            # Transform pc using gt
            matching_tf_stamped = TransformStamped()
            matching_tf_stamped.transform = gt[i]
            pc_gt = do_transform_cloud(current_pc[i], matching_tf_stamped)
            pc_gt_points = pc2.read_points(pc_gt, field_names=("x", "y"), skip_nans=True)

             # Plot after matching (target_pc transformed by gt)
            ax4 = fig.add_subplot(224)
            for point in pc_gt_points:
                ax4.scatter(point[0], point[1], c='b', marker='.',s=0.3)
            for point in target_points4:
                ax4.scatter(point[0], point[1], c='r', marker='x',s=0.3)

            x = round(gt[i].translation.x,4)
            y = round(gt[i].translation.y,4)
            q = (gt[i].rotation.x, gt[i].rotation.y,gt[i].rotation.z,gt[i].rotation.w)
            _,_,yaw = euler_from_quaternion(q)
            yaw = round(yaw,4)

            ax4.set_title('After Matching (GT) \n x:{} \n y:{} \n yaw:{}'.format(x,y,yaw))
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            
            
            plt.tight_layout()

            plt.savefig('/home/tanakrit-ubuntu/project_ws/src/ho_localization/plots/matching_plot_{}.png'.format(rospy.Time.now()))


if __name__ == '__main__':

    rospy.init_node('laser_matching')
    robot = LaserMatchingNode("/cloud_in", "/odom_lm")
    rospy.loginfo("Laser Matchong node started")

    rospy.spin()