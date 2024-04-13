#!/usr/bin/python3

import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg
import ros_numpy
import open3d as oddd

rospy.init_node("laserscan_to_pointcloud")

lp = lg.LaserProjection()

pc_pub = rospy.Publisher("/cloud_in", PointCloud2, queue_size=1)

def scan_cb(msg):
    pc2_msg = lp.projectLaser(msg)
    # Convert ROS PointCloud2 message to NumPy array
    print("Received point cloud message")
    cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_msg)
    pc_pub.publish(pc2_msg)
    # Create Open3D point cloud 
    pcd = oddd.geometry.PointCloud()
    pcd.points = oddd.utility.Vector3dVector(cloud_arr)

    # Save point cloud to PCD file
    oddd.io.write_point_cloud("room_scan2.pcd", pcd, write_ascii=False, compressed=False, print_progress=True)
    rospy.loginfo("Point cloud saved")

rospy.Subscriber("/scan", LaserScan, scan_cb, queue_size=1)
rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, scan_cb, queue_size=1)
rospy.spin()