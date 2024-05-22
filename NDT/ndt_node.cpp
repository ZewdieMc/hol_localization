#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Quaternion.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <ho_localization/PointCloudTransform.h>

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include <iostream>
#include <thread>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

using namespace std::chrono_literals;

std::stack<pcl::PointCloud<pcl::PointXYZ>> cloud_stack;//! Map
ros::Time last_time = ros::Time(0);
pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);

//! Set initial alignment estimate found using robot odometry.
Eigen::AngleAxisf init_rotation (0, Eigen::Vector3f::UnitZ ());
Eigen::Translation3f init_translation (0, 0.0, 0);
Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();
Eigen::Matrix4f final_tf;


int registeration ()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize (0.05, 0.05, 0.05);  //0.05
  approximate_voxel_filter.setInputCloud (input_cloud);
  approximate_voxel_filter.filter (*filtered_cloud);
  std::cout << "Filtered cloud contains " << filtered_cloud->size ()
            << " data points" << std::endl;

  // Initializing Normal Distributions Transform (NDT).
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  ndt.setTransformationEpsilon (0.00001);
  // Setting maximum step size for More-Thuente line search.
  ndt.setStepSize (0.03); //0.1
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
  ndt.setResolution (1); //

  // Setting max number of registration iterations.
  ndt.setMaximumIterations (500);

  // Setting point cloud to be aligned.
  ndt.setInputSource (filtered_cloud);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget (target_cloud);
  

  // Calculating required rigid transform to align the input cloud to the target cloud.
  ndt.align (*output_cloud, init_guess);

  // std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
  //           << " score: " << ndt.getFitnessScore () << std::endl;

  // Transforming unfiltered, input cloud using found transform.
  // pcl::transformPointCloud (*input_cloud, *output_cloud, ndt.getFinalTransformation ());

  // Saving transformed input cloud.
  // pcl::io::savePCDFileASCII ("room_scan2_transformed.pcd", *output_cloud);

  std::cout << ndt.getFinalTransformation () <<   std::endl;
  final_tf = ndt.getFinalTransformation();

  return (0);
}

void visualize(){
    // Initializing point cloud visualizer
  pcl::visualization::PCLVisualizer::Ptr
  viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer_final->setBackgroundColor (0, 0, 0);

  // Coloring and visualizing target cloud (red).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  target_color (target_cloud, 255, 0, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (target_cloud, target_color, "target cloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "target cloud");

  // Coloring and visualizing transformed input cloud (green).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  output_color (output_cloud, 0, 255, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (output_cloud, output_color, "output cloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "output cloud");

  // Starting visualizer
  viewer_final->addCoordinateSystem (1.0, "global");
  viewer_final->initCameraParameters ();

  // Wait until visualizer window is closed.
  while (!viewer_final->wasStopped ())
  {
    viewer_final->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
}

void cloud_callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
  int init_guess[] = {0, 0, 0, 0};
  //save first scan as target
  if (last_time.isZero())
  {
    pcl::fromROSMsg(*cloud_msg, *target_cloud);
    cloud_stack.push(*target_cloud);
  }
  else if(!last_time.isZero()){
    pcl::fromROSMsg(*cloud_msg, *input_cloud);
    cloud_stack.push(*input_cloud);
    registeration();
    // Update target cloud for next registration
    *target_cloud = *input_cloud;
  }
  last_time = ros::Time::now();
}

bool matching_service(ho_localization::PointCloudTransform::Request  &req,
         ho_localization::PointCloudTransform::Response &res)
{
  // Translation
  double x, y;
  x = req.initial_guese.translation.x;
  y = req.initial_guese.translation.y;

  // Orientation to RPY
  tf2::Quaternion quat;
  tf2::convert(req.initial_guese.rotation, quat);  // Convert to tf2 quaternion
  double roll, pitch, yaw;
  tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);  // Extract RPY angle

  // initial guese 
  Eigen::AngleAxisf init_rotation (yaw, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (x, y, 0.0);
  init_guess = (init_translation * init_rotation).matrix ();

  // pc2 
  sensor_msgs::PointCloud2ConstPtr cloud_msg = boost::make_shared<const sensor_msgs::PointCloud2>(req.target);
  pcl::fromROSMsg(*cloud_msg, *target_cloud);
  sensor_msgs::PointCloud2ConstPtr cloud_msg2 = boost::make_shared<const sensor_msgs::PointCloud2>(req.current);
  pcl::fromROSMsg(*cloud_msg2, *input_cloud);

  // start matching process
  registeration();

  // Response Final Transform
  Eigen::Vector3f translation = final_tf.block<3, 1>(0, 3);
  Eigen::Matrix3f rotationMatrix = final_tf.block<3, 3>(0, 0);
  Eigen::Quaternionf rotationQuat(rotationMatrix);

  // Create a geometry_msgs::Transform message
  geometry_msgs::Transform transformMsg;
  
  // Set the translation components
  transformMsg.translation.x = translation.x();
  transformMsg.translation.y = translation.y();
  transformMsg.translation.z = translation.z();

  // Set the rotation components (quaternion)
  transformMsg.rotation.x = rotationQuat.x();
  transformMsg.rotation.y = rotationQuat.y();
  transformMsg.rotation.z = rotationQuat.z();
  transformMsg.rotation.w = rotationQuat.w();

  res.transform = transformMsg;

  return true;
}

int main(int argc, char **argv){
  ros::init(argc, argv, "ndt_node");
  ros::NodeHandle nh;

  // ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_in", 1, cloud_callback);
  ros::ServiceServer service = nh.advertiseService("ndt_matching", matching_service);  
  // ros::Rate loop_rate();
  while(ros::ok()){
    ros::spinOnce();
    // loop_rate.sleep();
  }

  // visualize();
  return 0;
}
