import open3d as o3d

def convert_annotations_to_point_cloud(annotations):
    """
    Convert user point annotations to Open3D point cloud format.

    Args:
        annotations (list): List of user point annotations.

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud object.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(annotations)
    return point_cloud

def save_point_cloud(point_cloud, filename):
    """
    Save Open3D point cloud to a file.

    Args:
        point_cloud (o3d.geometry.PointCloud): Open3D point cloud object.
        filename (str): Name of the file to save the point cloud.
    """
    o3d.io.write_point_cloud(filename, point_cloud)