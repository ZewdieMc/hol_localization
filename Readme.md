# Robot Localization Module - ROS Package

This ROS package is designed to handle the robot localization module, comprising two primary nodes responsible for localization and transformation calculations.

## Setup and Usage

1. **Run the Simulation**
   - Launch the simulation environment by running the simulation launch file:
     ```sh
     roslaunch ho_localization simulation.launch
     ```
   - This starts the Stonefish simulation.

2. **Run the Localization Module**
   - Activate all the required modules by running the localization launch file:
     ```sh
     roslaunch ho_localization localization.launch
     ```

3. **Visualize Using RViz**
   - Open RViz with the provided configuration file:
     ```sh
     rviz -d $(rospack find ho_localization)/rviz/hol.rviz
     ```

4. **Control the Robot**
   - Publish velocity commands to the robot using any kind of publisher. The commands should be sent to the `/cmd_vel` topic. For example:
     ```sh
     rosrun teleop_twist_keyboard teleop_twist_keyboard.py
     ```

## Main Nodes

### 1. `localization_node`

- **Purpose**: Publishes the robot's transformation in the world frame and also the map in ther form of combined pointcloud.
- **Method**: Utilizes PBEKFSLAM (Particle-Based Extended Kalman Filter Simultaneous Localization and Mapping) to estimate the transformation and building the map.

#### Publishers

1. **Odometry Data**
   - **Topic**: `/odom`
   - **Message Type**: `Odometry`
   - **Queue Size**: 1
   - **Description**: Publishes odometry data.

2. **Pure Odometry Data**
   - **Topic**: `/odom_pure`
   - **Message Type**: `Odometry`
   - **Queue Size**: 1
   - **Description**: Publishes pure odometry data without any corrections from NDT (Pure EKF).

3. **Combined Point Cloud Data**
   - **Topic**: `/combined_pc`
   - **Message Type**: `PointCloud2`
   - **Queue Size**: 1
   - **Description**: Publishes combined point cloud data.

4. **Prediction Point Cloud Data**
   - **Topic**: `/prediction_pc`
   - **Message Type**: `PointCloud2`
   - **Queue Size**: 1
   - **Description**: Publishes combined point cloud data from pure EKF.

5. **Visualization States**
   - **Topic**: `~states`
   - **Message Type**: `MarkerArray`
   - **Queue Size**: 1
   - **Description**: Publishes visualization markers for states and theirs corresponding covariances.

#### Subscribers

1. **Point Cloud Data**
   - **Topic**: `/cloud_in`
   - **Message Type**: `PointCloud2`
   - **Callback Function**: `self.pc_cb`
   - **Description**: Subscribes to point cloud data.

2. **Joint State Data**
   - **Topic**: `/turtlebot/joint_states`
   - **Message Type**: `JointState`
   - **Callback Function**: `self.joint_state_callback`
   - **Description**: Subscribes to joint state data, providing wheel angular velocities.

3. **Ground Truth Odometry Data**
   - **Topic**: `/turtlebot/kobuki/ground_truth`
   - **Message Type**: `Odometry`
   - **Callback Function**: `self.gt_callback`
   - **Description**: Subscribes to ground truth odometry data.

4. **IMU Data**
   - **Topic**: `turtlebot/kobuki/sensors/imu`
   - **Message Type**: `Imu` 
   - **Callback Function**: `self.imu_callback`
   - **Description**: Subscribes to IMU data for orientation and angular velocity information.

### 2. `NDT_node`

- **Purpose**: Calculates the scan matching transformation.
- **Method**: Employs NDT (Normal Distributions Transform) for scan matching to determine the transformation.

#### Service

1. **NDT Matching Service**
   - **Service Name**: `ndt_matching`
   - **Service Type**: `PointCloudTransform`
     - **Request**:
       - **initial_guess**: `geometry_msgs/Transform`
       - **target**: `sensor_msgs/PointCloud2`
       - **current**: `sensor_msgs/PointCloud2`
     - **Response**:
       - **transform**: `geometry_msgs/Transform`
   - **Function**: `matching_service`
   - **Description**: Provides a service for NDT-based scan matching, accepting an initial guess and two point clouds (target and current), and returning the computed transformation.