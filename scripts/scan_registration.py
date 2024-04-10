import open3d as o3d
import numpy as np

def ndt_registration(target_cloud, input_cloud):
    # Convert point clouds to open3d format
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(np.asarray(target_cloud.points))

    input_o3d = o3d.geometry.PointCloud()
    input_o3d.points = o3d.utility.Vector3dVector(np.asarray(input_cloud.points))

    # Perform NDT registration
    voxel_size = 0.05
    target_down = target_o3d.voxel_down_sample(voxel_size)
    input_down = input_o3d.voxel_down_sample(voxel_size)

    o3d.registration.set_4d_extension(False)
    criteria = o3d.registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = 500
    criteria.relative_fitness = 1e-6
    criteria.relative_rmse = 1e-6

    result = o3d.registration.registration_colored_icp(
        input_down, target_down, voxel_size, np.eye(4), criteria
    )

    # Get registered point cloud
    input_transformed = input_down.transform(result.transformation)
    registered_cloud = input_transformed.voxel_down_sample(voxel_size)

    return registered_cloud

def load_point_cloud(file_path):
    # Load point cloud from file
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

if __name__ == "__main__":
    # Load target and input point clouds
    target_cloud = load_point_cloud("target_cloud.pcd")
    input_cloud = load_point_cloud("input_cloud.pcd")

    # Perform NDT registration
    registered_cloud = ndt_registration(target_cloud, input_cloud)

    # Save registered point cloud to file (optional)
    o3d.io.write_point_cloud("registered_cloud.pcd", registered_cloud)

