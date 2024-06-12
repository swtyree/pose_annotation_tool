from pathlib import Path

import numpy as np
import open3d as o3d


# Load RGB-D image as Open3D point cloud
def load_rgbd_pcd(
    rgb_fn: Path,
    depth_fn: Path,
    camera_info: dict,
) -> o3d.geometry.PointCloud:
    # Read RGB and depth images
    color_raw = o3d.io.read_image(str(rgb_fn))
    depth_raw = o3d.io.read_image(str(depth_fn))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=camera_info[
            "depth_scale"
        ],  # Open3D actually returns depth in meters
        convert_rgb_to_intensity=False,
    )

    # Convert to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            camera_info["width"],
            camera_info["height"],
            camera_info["cam_K"][0, 0],
            camera_info["cam_K"][1, 1],
            camera_info["cam_K"][0, 2],
            camera_info["cam_K"][1, 2],
        ),
    )

    # Scale point cloud to mm
    pcd.scale(1000.0, center=np.zeros(3))
    return pcd


# Load mesh as Open3D point cloud
def load_ply_pcd(
    mesh_fn: Path,
    compute_normals: bool = False,
) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(mesh_fn.as_posix())
    if compute_normals:
        pcd.estimate_normals()
    return pcd


# Convert numpy array to Open3D point cloud
def numpy_2_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# Reset points in Open3D point cloud
def reset_pcd(dest_pcd: o3d.geometry.PointCloud, source_pcd: o3d.geometry.PointCloud):
    dest_points = np.asarray(dest_pcd.points)
    source_points = np.asarray(source_pcd.points)
    assert (
        dest_points.shape == source_points.shape
    ), "Point clouds must have the same shape"
    dest_points[:, :] = source_points[:, :]


# Get scale from 4x4 transformation matrix
def get_scale_from_transform(transform: np.ndarray) -> tuple[float, np.ndarray]:
    transform_copy = transform.copy()
    scale = np.linalg.norm(transform_copy[:3, :3], ord=2)
    transform_copy[:3, :3] /= scale
    return scale, transform_copy


# Decompose 4x4 transformation matrix into translation, rotation, and scale
def decompose_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scale, Rt = get_scale_from_transform(transform)
    rotation = Rt[:3, :3]
    translation = Rt[:3, 3]
    return rotation, translation, scale


# Print 4x4 transformation matrix in human-readable format
def print_transform(transform: np.ndarray, title=None):
    rotation, translation, scale = decompose_transform(transform)
    if title is not None:
        print(title)
    print(f"Translation: {translation}")
    print(f"Rotation:\n{rotation}")
    print(f"Scale: {scale}")
    print()
