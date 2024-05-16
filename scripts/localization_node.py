#!/usr/bin/python3

import rospy
from sensor_msgs.msg import JointState, Imu
from  nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Transform
import numpy as np
from numpy import cos, sin
import tf.msg
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf
from dead_reckoning import DeadReckoning
from PEKFSLAM import PEKFSLAM
from utils.ekf_utils import pose_prediction, wrap_angle, oplus ,ominus
from ho_localization.srv import OdomTransform, OdomTransformRequest, OdomTransformResponse
from ho_localization.srv import OdomTransform, PointCloudTransform, PointCloudTransformRequest

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
import scipy
import rosparam
class LocalizationNode:
    def __init__(self, joint_state_topic, odom_topic,imu_topic,pc_topic, Wb=0.235, Wr=0.035):
        self.map_enable = rospy.get_param("enable_map",default="True")

        self.scan_frame = "turtlebot/kobuki/rplidar"
        self.bf_frame = "turtlebot/kobuki/base_footprint"
        self.world_frame = "world_ned"
        # Dead Reckoning Module
        self.dr = DeadReckoning(Wb, Wr)
        # wheel joint names
        self.left_wheel_name = "turtlebot/kobuki/wheel_left_joint"
        self.right_wheel_name = "turtlebot/kobuki/wheel_right_joint"
        # flags
        self.left_vel_arrived = False
        self.right_vel_arrived = False
        
        # Filter Module
        self.xk_0 = np.array([3.0, -0.78, np.pi/2]).reshape(-1,1)
        #self.xk_0 = np.array([0.0, 0.0,0.0]).reshape(-1,1)

        self.Pk_0 = np.array([[0.000, 0 ,0],
                            [0, 0.0000, 0],
                            [0, 0 ,0.000]])
        self.lm_cov = np.array([[0.05**2, 0 ,0],
                            [0, 0.05**2, 0],
                            [0, 0 ,0.005**2]])
        
        self.filter = PEKFSLAM(self.xk_0, self.Pk_0)

        self.map = [] # contain [PoseCovStamped, PC2]
        self.map_flag = [] # flag for combining
        self.map_state = []
        self.map_gt = [] # for testing only
        self.current_pc = None
        self.prediction_pc = None
        self.combined_pc = None
        self.gt = None

        self.map_update_interval = 4
        self.overlap_threshold = 1

        # For initial guese calculation
        self.previous_pose = self.filter.xk[-3:] 
        self.previous_gt = self.previous_pose
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        # Joint State Publisher
        self.joint_state_pub = rospy.Publisher(joint_state_topic, JointState, queue_size=1)
        # ODOM PUBLISHER
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=1)
        self.combined_pc_pub = rospy.Publisher("/combined_pc", PointCloud2, queue_size=1)
        self.prediction_pc_pub = rospy.Publisher("/prediction_pc", PointCloud2, queue_size=1)

        self.visualize_pub = rospy.Publisher("~states", MarkerArray, queue_size=1)

        # Subscriber
        # PC SUBSCRIBER
        self.pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_cb)
        # JOINT STATE SUBSCRIBER #! Gives the wheel angular velocities
        self.joint_state_sub = rospy.Subscriber(joint_state_topic, JointState, self.joint_state_callback)
        self.gt_sub = rospy.Subscriber("/turtlebot/kobuki/ground_truth", Odometry, self.gt_callback)
        # IMU SUBSCRIBER
        self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback)


        #Services
        # Initial Guese service
        self.initial_srv = rospy.Service('get_initial_guess', OdomTransform, self.handle_get_initial_guess)
        # Gt service
        self.gt_srv = rospy.Service('get_gt', OdomTransform, self.handle_get_gt)

        self.tf_bc = tf2_ros.TransformBroadcaster()

        rospy.sleep(5)
        rospy.Timer(rospy.Duration(0.01), self.odom_msg_pub)
        rospy.sleep(5)
        # Timer
        self.initialize_map()
        rospy.sleep(1)

        if self.map_enable == "True":
            rospy.loginfo("Map is enabled")
        
            rospy.Timer(rospy.Duration(self.map_update_interval), self.main_loop)
            rospy.Timer(rospy.Duration(0.5), self.publish_map)
            rospy.Timer(rospy.Duration(0.5), self.visualize_states)
        


    ##################################################################
    #### Callback functions
    ##################################################################
    def joint_state_callback(self, data):
        '''
        callback for /joint_state 
            - assign left and right velocity
            - calculate displacement using deadreckoning
            - do prediction
        '''
        if data.name[0] == self.left_wheel_name:
            self.left_vel = data.velocity[0]
            self.left_vel_arrived = True
            self.right_vel = data.velocity[1]
            self.right_vel_arrived = True

        if data.name[0] == self.right_wheel_name:
            self.right_vel = data.velocity[0]
            self.right_vel_arrived = True
            
        if self.left_vel_arrived and self.right_vel_arrived:
            
            # set flags
            self.left_vel_arrived = False
            self.right_vel_arrived = False

            uk, Qk, dt = self.dr.get_input(self.left_vel, self.right_vel)
            self.filter.dt = dt
            xk_bar, Pk_bar = self.filter.prediction(uk, Qk)

    def imu_callback(self, data):
        '''
        callback for /imu
        '''
        q = [data.orientation.x,data.orientation.y,data.orientation.z,data.orientation.w]
        euler = tf.transformations.euler_from_quaternion(q)
        
        zk = np.array([wrap_angle(-euler[2])]).reshape(-1,1)
        Rk = np.array([0.1]).reshape(-1,1)

        self.filter.update_imu(zk,Rk)

    def gt_callback(self,data):
        q = [data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w]
        _,_,yaw = euler_from_quaternion(q)
        gt_pose = np.array([data.pose.pose.position.x,  data.pose.pose.position.y,yaw]).reshape(-1,1)
        self.gt = gt_pose

    def pc_cb(self, data):
        '''
        callback for point cloud 
            
        '''
        # Transform point cloud to reference with base_footprint
        pc = self.transform_pc(data, self.bf_frame)
        # pc = data
        self.current_pc = pc 

        self.current_timestamp = data.header.stamp
    

    def handle_get_initial_guess(self, req):
        rospy.logwarn("Handle get_initial_guess")
        self.current_pose = self.filter.xk[:-3]
        dis = oplus(ominus(self.previous_pose), self.current_pose)
        # claculate displacement
        dx = dis[0,0]
        dy = dis[1,0]
        dyaw = dis[2,0]
        q = quaternion_from_euler(0, 0, float(dyaw))

        # service response
        res = OdomTransformResponse()
        res.transform.translation.x = dx
        res.transform.translation.y = dy
        res.transform.translation.z = 0
        res.transform.rotation.x = q[0]
        res.transform.rotation.y = q[1]
        res.transform.rotation.z = q[2]
        res.transform.rotation.w = q[3]

        # Update previous stamp
        self.previous_pose = self.current_pose
        
        rospy.logwarn("Initial Guese: {}".format(res.transform))
        return res
    
    def handle_get_gt(self, req):
        rospy.logwarn("Handle get_gt")

        current_gt = self.gt

        dis = oplus(ominus(self.previous_gt), current_gt)
        # claculate displacement
        dx = dis[0,0]
        dy = dis[1,0]
        dyaw = dis[2,0]
        q = quaternion_from_euler(0, 0, float(dyaw))

        # service response
        res = OdomTransformResponse()
        res.transform.translation.x = dx
        res.transform.translation.y = dy
        res.transform.translation.z = 0
        res.transform.rotation.x = q[0]
        res.transform.rotation.y = q[1]
        res.transform.rotation.z = q[2]
        res.transform.rotation.w = q[3]

        # Update previous stamp
        self.previous_gt = current_gt
        
        rospy.logwarn("Ground Truth: {}".format(res.transform))
        return res
    
    def get_scan_matching(self, current_pc, target_pc, initial_guess_tf):
        '''
        function to call /ndt_matching service
        '''
        rospy.logwarn("Calling get_matching_tf...")
        rospy.wait_for_service('ndt_matching')
        try:
            get_matching_tf = rospy.ServiceProxy('ndt_matching', PointCloudTransform)
            req = PointCloudTransformRequest()
            req.initial_guese = initial_guess_tf
            req.target = target_pc
            req.current = current_pc
            resp = get_matching_tf(req)
            # rospy.logwarn("matching: {}".format(resp.transform))
            
            dx = resp.transform.translation.x
            dy = resp.transform.translation.y
            q = (resp.transform.rotation.x, resp.transform.rotation.y, resp.transform.rotation.z, resp.transform.rotation.w)
            euler = euler_from_quaternion(q)
            lm_matching = np.array([dx,dy,euler[2]]).reshape(-1,1)

            lm_matching_cov = self.lm_cov
            

            return resp.transform, lm_matching, lm_matching_cov
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return None, None, None

    def get_gt_matching(self, previous_gt, current_gt):
        

        dis = oplus(ominus(previous_gt), current_gt)
        # claculate displacement
        dx = dis[0,0]
        dy = dis[1,0]
        dyaw = wrap_angle(dis[2,0])
        q = quaternion_from_euler(0, 0, float(dyaw))

        # service response
        gt_tf = Transform()
        gt_tf.translation.x = dx
        gt_tf.translation.y = dy
        gt_tf.translation.z = 0
        gt_tf.rotation.x = q[0]
        gt_tf.rotation.y = q[1]
        gt_tf.rotation.z = q[2]
        gt_tf.rotation.w = q[3]

        gt =  np.array([dx,dy,dyaw]).reshape(-1,1)
        gt_cov = self.lm_cov
        return gt_tf, gt , gt_cov
        

    ##################################################################
    #### Timer Functions
    ##################################################################
    def odom_msg_pub(self,_):
        '''
        Publish /odom and TF using current state vector
        '''
        odom = Odometry()
        odom.header.stamp = self.current_timestamp 
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.bf_frame

        odom.pose = self.get_robot_pose_cov(self.filter.xk, self.filter.Pk).pose

        self.odom_pub.publish(odom)

        q = quaternion_from_euler(0, 0, float(self.filter.xk[-1, 0]))
        # tf.TransformBroadcaster().sendTransform((float(self.filter.xk[-3, 0]), float(self.filter.xk[-2, 0]), 0.0), q, rospy.Time.now(
        # ), odom.child_frame_id, odom.header.frame_id)
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.header.stamp = self.current_timestamp
        tf_msg.child_frame_id = self.bf_frame

        tf_msg.transform.translation.x = float(self.filter.xk[-3, 0])
        tf_msg.transform.translation.y = float(self.filter.xk[-2, 0])
        tf_msg.transform.translation.z = 0
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        self.tf_bc.sendTransform(tf_msg)
    
    def odom_msg_gt_pub(self,_):
        '''
        Publish /odom and TF using current state vector
        '''
        odom = Odometry()
        odom.header.stamp = self.current_timestamp
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.bf_frame

        odom.pose = self.get_robot_pose_cov(self.filter.xk, self.filter.Pk).pose

        self.odom_pub.publish(odom)

        q = quaternion_from_euler(0, 0, float(self.gt[-1, 0]))
        # tf.TransformBroadcaster().sendTransform((float(self.gt[-3, 0]), float(self.gt[-2, 0]), 0.0), q, rospy.Time.now(
        # ), odom.child_frame_id, odom.header.frame_id)
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.child_frame_id = self.bf_frame

        tf_msg.transform.translation.x = float(self.gt[-3, 0])
        tf_msg.transform.translation.y = float(self.gt[-2, 0])
        tf_msg.transform.translation.z = 0
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        self.tf_bc.sendTransform(tf_msg)


    def main_loop(self,_):
        '''
        Main PEKF update loop
        '''
        
        current_pc = self.current_pc

        if self.filter.need_augment():
            xk_plus,Pk_plus = self.filter.augment_state() # update state

            current_pc = self.current_pc
            self.map.append(current_pc) # update map
            self.map_flag.append(False)
            self.map_gt.append(self.gt)

            new_state = xk_plus[-6:-3,0].reshape(-1,1)  
            new_state_cov = Pk_plus[-6:-3,-6:-3]
            
            self.map_state.append(new_state)

            # find overlapping scan 
            overlaped_list = self.find_overlapping(xk_plus, new_state)

            if len(self.map) != (self.filter.xk.shape[0]//3)-1:
                rospy.logerr("state is unbalncaned xk:{}, map:{}".format((self.filter.xk.shape[0]//3)-1, len(self.map)))

            # find laser matching tf for each overlapping scan
            zlm = np.zeros((0,1))
            Rlm = np.zeros((0,0))
            hlm = np.zeros((0,1))
            Plm = np.zeros((0,0))
            for idx,i  in enumerate(overlaped_list):
                hypo_state = xk_plus[i*3:i*3+3,0].reshape(-1,1)
                target_pc = self.map[i]
                
                initial_guess_tf ,initial_guess, initial_guess_cov = self.filter.find_initial_guess(new_state,hypo_state, new_state_cov, Pk_plus[i*3:i*3+3,i:i+3])
                _, scan_matching, scan_matching_cov = self.get_scan_matching(target_pc,current_pc,initial_guess_tf)

                if self.filter.individual_compatable(scan_matching,initial_guess,scan_matching_cov,initial_guess_cov):
                    zlm = np.block([[zlm], [scan_matching]])
                    Rlm = scipy.linalg.block_diag(Rlm, scan_matching_cov)
                    hlm = np.block([[hlm], [initial_guess]])
                    Plm = scipy.linalg.block_diag(Plm, initial_guess_cov)
                    self.map_flag[i] = False
                else:
                    overlaped_list.pop(idx)
            if len(overlaped_list) > 0:
                self.filter.update_lm(zlm,Rlm,hlm,overlaped_list)

    def main_loop_gt(self,_):
        '''
        Main PEKF update loop
        '''
        
        
        if self.filter.need_augment():
            
            xk_plus,Pk_plus = self.filter.augment_state() # update state

            current_pc = self.current_pc
            self.map.append(current_pc) # update map
            self.map_flag.append(False)
            self.map_gt.append(self.gt)
            new_state = xk_plus[-6:-3,0].reshape(-1,1)  
            new_state_cov = Pk_plus[-6:-3,-6:-3]
            
            # find overlapping scan 
            overlaped_list = self.find_overlapping(xk_plus, new_state)
            
            if len(self.map) != (self.filter.xk.shape[0]//3)-1:
                rospy.logerr("state is unbalncaned")
            # find laser matching tf for each overlapping scan
            zlm = np.zeros((0,1))
            Rlm = np.zeros((0,0))
            hlm = np.zeros((0,1))
            Plm = np.zeros((0,0))
            for i in overlaped_list:
                
                target_pc = self.map[i]
                hypo_state = xk_plus[i*3:i*3+3,0].reshape(-1,1)
                initial_guess_tf ,initial_guess, initial_guess_cov = self.filter.find_initial_guess(new_state,hypo_state, new_state_cov,Pk_plus[i*3:i*3+3,i:i+3])

                _, scan_matching, scan_matching_cov = self.get_gt_matching(self.map_gt[-1],self.map_gt[i])

                if self.filter.individual_compatable(scan_matching,initial_guess,scan_matching_cov,initial_guess_cov):
                    zlm = np.block([[zlm], [scan_matching]])
                    Rlm = scipy.linalg.block_diag(Rlm, scan_matching_cov)
                    hlm = np.block([[hlm], [initial_guess]])
                    Plm = scipy.linalg.block_diag(Plm, initial_guess_cov)
                    self.map_flag[i] = False
                else:
                    overlaped_list.pop(i)
            if len(overlaped_list) > 0:
                self.filter.update_lm(zlm,Rlm,hlm,overlaped_list)
                    


    def find_overlapping(self,states ,new_state):
        '''
        return the index of the overlapping scan in the map
        '''
        # for now use distance threshold
        # rospy.logwarn("Vp: {}".format(((states.shape[0])//3)-1))
        overlaped_list = []
        for i in range(0,(states.shape[0]-6)//3):
            dis = np.linalg.norm(states[i*3:i*3+2,0] - new_state[:2,0])
            if dis < self.overlap_threshold:
                overlaped_list.append(i)
        return overlaped_list
            
              

    def initialize_map(self):
        '''
        Initialize map with first point cloud
        '''
        while not self.current_pc  and  not rospy.is_shutdown():
            rospy.loginfo("Waiting for  first pc2...")
            rospy.sleep(0.1)

        current_pc = self.current_pc
        self.map.append(current_pc)      
        self.map_flag.append(False)
        self.map_gt.append(self.gt)      
        self.filter.augment_state()
        self.map_state.append(self.filter.xk[-6:-3,0].reshape(-1,1))


        # self.combined_pc = self.transform_pc(current_pc, self.world_frame)
    
    ##################################################################
    #### Visualization functions
    ##################################################################
    def publish_map(self,_):
        try:
            if len(self.map) > 0 and self.filter.xk.shape[0] > 3:
                xk = self.filter.xk.copy()
                map = self.map
                self.combined_pc = self.transform_pc_from_pose(map[0],xk[0:3,0].reshape(-1,1) )
                self.prediction_pc = self.transform_pc_from_pose(map[0],self.map_state[0]) 

                for i in range(1,(xk.shape[0]//3)-1):
                    # if not map_flag[i]:
                    # rospy.loginfo("Merge new pc...")
                    # map_flag[i] = True
                    
                    new_prediction_pc = self.transform_pc_from_pose(map[i],self.map_state[i]) 
                    self.prediction_pc = self.merge_pointclouds(self.prediction_pc ,new_prediction_pc)


                    vp = xk[i*3:i*3+3,0].reshape(-1,1)
                    new_pc = self.transform_pc_from_pose(map[i],vp )
                    self.combined_pc = self.merge_pointclouds(self.combined_pc, new_pc)
                
                self.combined_pc_pub.publish(self.combined_pc)
                self.prediction_pc_pub.publish(self.prediction_pc)
        except Exception as e:
            rospy.logerr(e)

    def visualize_states(self,_):
        try:
            states = self.filter.xk
            covariances = self.filter.Pk
            marker_array_msg = MarkerArray()
            if len(self.map) > 0 and self.filter.xk.shape[0] > 3 :
                xk = self.filter.xk.copy()
                map = self.map
                for i in range(0, (xk.shape[0]//3)-1): # prooved
                    state = states[i*3:i*3+3,0]
                    cov   =  covariances[i*3:i*3+2, i*3:i*3+2]

                    state_marker = self.create_state_marker(state,i)
                    cov_marker = self.create_cov_marker(cov,state,i)

                    marker_array_msg.markers.append(state_marker)
                    marker_array_msg.markers.append(cov_marker)


                    # cov_current = self.filter.Pk[-3:-1,-3:-1]
                    # print(cov_current)
                    # poscov = self.get_robot_pose_cov(self.filter.xk, self.filter.Pk)
                    # print(poscov.pose.covariance)
                    # cov_current_marker = self.create_cov_marker(cov_current,state,i)
                    # cov_current_marker.color.r = 0
                    # cov_current_marker.color.g = 0
                    # cov_current_marker.color.b = 1
                    # cov_current_marker.color.a = 0.5
                    # marker_array_msg.markers.append(cov_current_marker)
            
            self.visualize_pub.publish(marker_array_msg)
        except Exception as e:
            rospy.logerr(e)

        

    def create_state_marker(self, state, idx = 0):
        state_marker = Marker()
        state_marker.header.stamp = self.current_timestamp
        state_marker.header.frame_id = self.world_frame
        state_marker.ns = "states"
        state_marker.id = idx
        state_marker.type = Marker.ARROW
        state_marker.action = Marker.ADD
        state_marker.pose.position.x = state[0]
        state_marker.pose.position.y = state[1]
        state_marker.pose.position.z = 0
        q = quaternion_from_euler(0, 0, state[2])

        state_marker.pose.orientation.x = q[0]
        state_marker.pose.orientation.y = q[1]
        state_marker.pose.orientation.z = q[2]
        state_marker.pose.orientation.w = q[3]
        state_marker.scale.x = 0.5
        state_marker.scale.y = 0.05
        state_marker.scale.z = 0.05
        r,g,b = self.scale_rgb_by_index(idx)
        state_marker.color.a = 1.0
        state_marker.color.r = r
        state_marker.color.g = g
        state_marker.color.b = b

        return state_marker


    def create_cov_marker(self,cov,state,idx):
        cov_marker = Marker()
        cov_marker.header.stamp = self.current_timestamp
        cov_marker.header.frame_id = self.world_frame
        cov_marker.ns = "covs"
        cov_marker.id = idx
        cov_marker.type = Marker.SPHERE
        cov_marker.action = Marker.ADD
        cov_marker.pose.position.x = state[0]
        cov_marker.pose.position.y = state[1]
        cov_marker.pose.position.z = 0
        if cov[1,0] <= 0.0001 and  cov[1,0] >= -0.0001:
            cov[1,0] = 0.
        if cov[0,1] <= 0.0001 and  cov[1,0] >= -0.0001:
            cov[0,1] = 0.000
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Find major and minor axes lengths
        x_axis_length = np.sqrt(eigenvalues[0])
        y_axis_length = np.sqrt(eigenvalues[1])
        # Find orientation of the ellipse
        orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        q = quaternion_from_euler(0, 0, orientation)

        cov_marker.pose.orientation.x = q[0]
        cov_marker.pose.orientation.y = q[1]
        cov_marker.pose.orientation.z = q[2]
        cov_marker.pose.orientation.w = q[3]
        cov_marker.scale.x = x_axis_length * 2 
        cov_marker.scale.y = y_axis_length * 2 
        cov_marker.scale.z = 0.01
        r,g,b = self.scale_rgb_by_index(idx)
        cov_marker.color.a = 1.0
        cov_marker.color.r = r
        cov_marker.color.g = g
        cov_marker.color.b = b

        return cov_marker


            
        

    ##################################################################
    #### Utility functions
    ##################################################################
    def get_robot_pose_cov(self, xk, Pk):
        '''
        extract robot pose from state vector
        '''
        robot_pose = PoseWithCovarianceStamped()
        robot_pose.header.stamp = rospy.Time.now()
        robot_pose.header.frame_id = self.world_frame

        robot_pose.pose.pose.position.x = xk[-3,0]
        robot_pose.pose.pose.position.y = xk[-2,0]
        robot_pose.pose.pose.position.z = 0

        q = quaternion_from_euler(0, 0, float(xk[-1, 0]))

        robot_pose.pose.pose.orientation.x = q[0]
        robot_pose.pose.pose.orientation.y = q[1]
        robot_pose.pose.pose.orientation.z = q[2]
        robot_pose.pose.pose.orientation.w = q[3]

        robot_pose.pose.covariance = list(np.array([[Pk[-3, -3], Pk[-3, -2], 0, 0, 0, Pk[-3, -1]],
                                [Pk[-2, -3], Pk[-2,-2], 0, 0, 0, Pk[-2, -1]],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [Pk[-1, -3], Pk[-1, -2], 0, 0, 0, Pk[-1, -1]]]).flatten())

        return robot_pose

    def transform_pc(self, pc_msg, target_frame):
        source_frame = pc_msg.header.frame_id

        # Wait for the transform to be available
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
        pc_transformed = do_transform_cloud(pc_msg, transform)
        return pc_transformed
    
    def transform_pc_from_pose(self, pc_msg, pose):
        transform = TransformStamped()
        transform.header.stamp = self.current_timestamp
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.bf_frame
        transform.transform.translation.x = pose[0,0]
        transform.transform.translation.y = pose[1,0]
        
        q = quaternion_from_euler(0, 0, float(pose[2, 0]))
        transform.transform.rotation.x = q[0]  # Assuming no rotation for simplicity
        transform.transform.rotation.y = q[1] 
        transform.transform.rotation.z = q[2] 
        transform.transform.rotation.w = q[3] 
        pc_transformed = do_transform_cloud(pc_msg, transform)
        return pc_transformed
    
    # def is_scene_change(self, current_pose):
    #     """
    #     check if current view is changed from lastest view in map
    #     """
    #     if self.map:
    #         x1 = current_pose.pose.pose.position.x
    #         y1 = current_pose.pose.pose.position.y
    #         yaw1 = self.calculate_yaw(current_pose)

    #         x2 = self.map[-1][0].pose.pose.position.x
    #         y2 = self.map[-1][0].pose.pose.position.y
    #         yaw2 = self.calculate_yaw(self.map[-1][0])
            
    #         # Calculating Euclidean distance between the two poses
    #         position_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #         angular_distance = wrap_angle(abs(yaw2-yaw1))

    #         if position_distance > 0.05 or angular_distance > 0.2:
    #             return True
    #         else:
    #             return False
                    
    #     else:
    #         return True
        
    def calculate_yaw(pose):
    # Extracting orientation data from PoseWithCovarianceStamped message
        quaternion = (
            pose.pose.pose.orientation.x,
            pose.pose.pose.orientation.y,
            pose.pose.pose.orientation.z,
            pose.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2] # Yaw angle in radians
        return yaw
    
    def merge_pointclouds(self,pc1, pc2):

        # Initialize combined PointCloud2 message
        combined_pc = PointCloud2()
        combined_pc.header = pc1.header
        combined_pc.header.stamp = self.current_timestamp
        combined_pc.height = 1
        combined_pc.width = pc1.width + pc2.width
        combined_pc.fields = pc1.fields
        combined_pc.is_bigendian = pc1.is_bigendian
        combined_pc.point_step = pc1.point_step
        combined_pc.row_step = combined_pc.point_step * combined_pc.width

        # Concatenate points from both point clouds
        # combined_pc_data = list(point_cloud2.read_points(pc1, field_names=("x", "y", "z"), skip_nans=True))
        # combined_pc_data.extend(point_cloud2.read_points(pc2, field_names=("x", "y", "z"), skip_nans=True))
        # # Convert combined point cloud data to a byte array
        # combined_pc_bytes = bytearray()
        # for point in combined_pc_data:
        #     combined_pc_bytes.extend(np.array(point).tobytes())
      
        
        combined_pc.data = pc1.data + pc2.data

        return combined_pc
    
    def scale_rgb_by_index(self,index):
        num_markers_per_channel = 10  # Number of markers per color channel
        
        # Determine which color channel to scale based on the index
        channel_index = index // num_markers_per_channel
        remainder = index % num_markers_per_channel
        
        # Initialize RGB values
        r = 0.0
        g = 0.0
        b = 0.0
        
        # Scale the appropriate color channel based on the remainder
        if channel_index == 0:  # Red channel
            r = float(remainder) / num_markers_per_channel
        elif channel_index == 1:  # Green channel
            g = float(remainder) / num_markers_per_channel
        else:  # Blue channel
            b = float(remainder) / num_markers_per_channel
        
        return r, g, b
    
    def find_initial_guess(self, previous_pose, current_pose):
        dis = oplus(ominus(previous_pose), current_pose)
        # claculate displacement
        dx = dis[0,0]
        dy = dis[1,0]
        dyaw = dis[2,0]
        q = quaternion_from_euler(0, 0, float(dyaw))

        # service response
        initial_guese_transform = Transform()
        initial_guese_transform.translation.x = dx
        initial_guese_transform.translation.y = dy
        initial_guese_transform.translation.z = 0
        initial_guese_transform.rotation.x = q[0]
        initial_guese_transform.rotation.y = q[1]
        initial_guese_transform.rotation.z = q[2]
        initial_guese_transform.rotation.w = q[3]

        return initial_guese_transform

if __name__ == '__main__':

    rospy.init_node('localization')
    robot = LocalizationNode("/turtlebot/joint_states", "/odom","/turtlebot/kobuki/sensors/imu_data","/cloud_in")
    rospy.loginfo("Localization node started")

    rospy.spin()