import argparse
from pathlib import Path

import bop_toolkit_lib
import bop_toolkit_lib.inout
import numpy as np
from loop_rate_limiters import RateLimiter


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meshes_dir",
        type=Path,
        required=True,
        help="Path to BOP meshes directory",
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        required=True,
        help="Path to BOP scene directory",
    )
    parser.add_argument(
        "--view_id",
        type=int,
        default=0,
        help="Scene view id to annotate",
    )
    parser.add_argument(
        "--instance_id",
        type=int,
        default=-1,
        help="Instance id to annotate (creates new instance if -1)",
    )
    parser.add_argument(
        "--object_id",
        type=str,
        default=None,
        help="Object id to annotate (required if creating new instance)",
    )
    parser.add_argument(
        "--optimize_scale",
        action="store_true",
        help="Optimize for scale",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Manually crop RGB-D point cloud",
    )
    parser.add_argument(
        "--rgbd_cache_dir",
        type=Path,
        default=None,
        help="RGBD cache directory to use (default: none)",
    )
    parser.add_argument(
        "--view_only",
        "--no_picker",
        action="store_true",
        help="View an existing annotation",
    )
    args = parser.parse_args()

    # Read or initialize scene GT
    scene_gt_fn = args.scene_dir / "scene_gt.json"
    if scene_gt_fn.exists():
        scene_gt = bop_toolkit_lib.inout.load_scene_gt(scene_gt_fn)
    else:
        scene_gt = {}

    # Check if we are annotating an existing instance
    if args.instance_id >= 0:
        # Check if the instance exists and matches expectations
        assert scene_gt_fn.exists(), f"Scene GT file {scene_gt_fn} does not exist"
        assert (
            args.view_id in scene_gt
        ), f"View {args.view_id} not in {scene_gt_fn}; unable to load existing instance"
        assert (
            len(scene_gt[args.view_id]) > args.instance_id
        ), f"Instance {args.instance_id} not found in {args.view_id} in {scene_gt_fn}"
        instance_dict = scene_gt[args.view_id][args.instance_id]
        if args.object_id is None:
            args.object_id = instance_dict["obj_id"]
        else:
            assert (
                instance_dict["obj_id"] == args.object_id
            ), f"Object ID {instance_dict['obj_id']} does not match {args.object_id}"

        # Set instance id and transform
        instance_id = args.instance_id
        current_transform = np.eye(4)
        current_transform[:3, :3] = instance_dict["cam_R_m2c"] * instance_dict.get(
            "__estimated_scale", 1
        )
        current_transform[:3, 3:] = instance_dict["cam_t_m2c"]
        loaded_transform = current_transform
        mesh_picked = np.array(
            instance_dict.get("__mesh_picked_points", np.zeros((0, 3)))
        )
        rgbd_picked = np.array(
            instance_dict.get("__rgbd_picked_points", np.zeros((0, 3)))
        )
        print(
            f"Loaded existing instance {instance_id} for object {args.object_id} from {scene_gt_fn}"
        )
    else:
        # Initialize new instance
        instance_id = len(scene_gt.setdefault(args.view_id, []))
        instance_dict = {}
        scene_gt[args.view_id].append(instance_dict)
        current_transform = np.eye(4)
        loaded_transform = current_transform
        mesh_picked = np.zeros((0, 3))
        rgbd_picked = np.zeros((0, 3))
        print(
            f"Creating new instance {instance_id} for object {args.object_id} in {scene_gt_fn}"
        )

    # Find necessary BOP files
    if args.object_id.isdigit():
        args.object_id = int(args.object_id)
        mesh_fn = args.meshes_dir / f"{args.object_id:06d}.ply"
    else:
        mesh_fn = args.meshes_dir / f"{args.object_id}.ply"
    assert mesh_fn.exists(), f"Mesh file {mesh_fn} does not exist"
    rgb_fn = args.scene_dir / "rgb" / f"{args.view_id:06d}.jpg"
    if not rgb_fn.exists():
        rgb_fn = rgb_fn.with_suffix(".png")
    assert rgb_fn.exists(), f"RGB-D file {rgb_fn.with_suffix('.*')} does not exist"
    depth_fn = args.scene_dir / "depth" / f"{args.view_id:06d}.png"
    assert depth_fn.exists(), f"RGB-D file {depth_fn} does not exist"
    scene_camera_fn = args.scene_dir / "scene_camera.json"
    assert (
        scene_camera_fn.exists()
    ), f"Scene camera file {scene_camera_fn} does not exist"

    # Read camera info
    scene_camera = bop_toolkit_lib.inout.load_scene_camera(scene_camera_fn)
    assert args.view_id in scene_camera, f"View {args.view_id} not in {scene_camera_fn}"
    camera_info = scene_camera[args.view_id]

    # Import modules (slow because of Open3D import)
    import open3d as o3d

    from src.registration import (
        ICPRegistrationHandler,
        PickedPointsHandler,
        Point2PointRegistrationHandler,
    )
    from src.utils import (
        decompose_transform,
        get_scale_from_transform,
        load_ply_pcd,
        load_rgbd_pcd,
        print_transform,
        reset_pcd,
    )
    from src.visualizer import (
        PickPointVisualizer,
        RegistrationVisualizer,
        ViewSettings,
        WindowSettings,
        crop_manually,
    )

    # Load mesh point cloud
    mesh_pcd = load_ply_pcd(mesh_fn)
    mesh_pcd_transformed = o3d.geometry.PointCloud(mesh_pcd)  # copy

    # Load RGBD point cloud
    rgbd_cache_fn = (
        args.rgbd_cache_dir / f"{args.view_id:06d}.ply" if args.rgbd_cache_dir else None
    )
    if rgbd_cache_fn and rgbd_cache_fn.exists():
        rgbd_pcd = load_ply_pcd(rgbd_cache_fn)
    else:
        rgbd_pcd = load_rgbd_pcd(rgb_fn, depth_fn, camera_info)

    # Crop RGBD point cloud, if requested
    if args.crop:
        crop_view_settings = ViewSettings(
            field_of_view=10.0,
            front=[-0.9, -0.3, -0.3],
            # lookat=[0.0, 0.0, rgbd_pcd.get_min_bound()[2]],
            lookat=rgbd_pcd.get_center().tolist(),
            up=[0.2, -0.9, 0.4],
            zoom=0.5,
        )
        rgbd_pcd = crop_manually(rgbd_pcd, crop_view_settings)

    # Cache RGBD point cloud
    if rgbd_cache_fn:
        if not rgbd_cache_fn.parent.exists():
            rgbd_cache_fn.parent.mkdir(parents=True)
        if not rgbd_cache_fn.exists() or args.crop:
            o3d.io.write_point_cloud(rgbd_cache_fn.as_posix(), rgbd_pcd)

    # Estimate normals for both point clouds
    radius = radius = 0.002 * 1000
    if not mesh_pcd.has_normals():
        mesh_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )
    if not rgbd_pcd.has_normals():
        rgbd_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )
        rgbd_pcd.orient_normals_towards_camera_location()

    # Set up visualizers
    window_settings = WindowSettings()
    mesh_window, _, rgbd_window = window_settings.hsplit([0.45, 0.1, 0.45])
    _, alignment_window, _ = window_settings.hsplit([0.15, 0.7, 0.15])
    mesh_window.name = "Mesh"
    rgbd_window.name = "RGBD"
    alignment_window.name = "Alignment"

    mesh_view_settings = ViewSettings(
        field_of_view=10.0,
        front=[1.0, 0.0, 0.0],
        lookat=mesh_pcd.get_center().tolist(),
        up=[0.0, 1.0, 0.0],
        zoom=1.0,
    )
    rgbd_view_settings = ViewSettings(
        field_of_view=10.0,
        front=[0.0, 0.0, -1.0],
        lookat=rgbd_pcd.get_center().tolist(),
        up=[0.0, -1.0, 0.0],
        zoom=0.85,
    )

    # Wrap visualizers in try-finally to ensure they are closed
    try:
        # Create visualizers
        if not args.view_only:
            mesh_point_picker = PickPointVisualizer(
                mesh_pcd, mesh_window, mesh_view_settings
            )
            rgbd_point_picker = PickPointVisualizer(
                rgbd_pcd, rgbd_window, rgbd_view_settings
            )
        alignment_vis = RegistrationVisualizer(
            alignment_window,
            rgbd_view_settings,
        )
        alignment_vis.add_geometry(mesh_pcd_transformed, reset_bounding_box=False)
        alignment_vis.add_geometry(rgbd_pcd, reset_bounding_box=True)

        # Create handlers for picked points and registration
        picked_points_mesh = (
            PickedPointsHandler(mesh_point_picker) if not args.view_only else None
        )
        picked_points_rgbd = (
            PickedPointsHandler(rgbd_point_picker) if not args.view_only else None
        )
        p2p_registration = Point2PointRegistrationHandler(
            with_scaling=args.optimize_scale
        )
        icp_registration = ICPRegistrationHandler()

        # Apply any existing transform to mesh point cloud
        mesh_pcd_transformed.transform(current_transform)

        # Run annotation/visualization loop
        rate = RateLimiter(frequency=60, warn=False)
        while alignment_vis.render_frame():
            # Check for updates to picked points and run point-to-point registration
            if not args.view_only and (
                picked_points_mesh.update() or picked_points_rgbd.update()
            ):
                # Get picked points
                mesh_picked = picked_points_mesh.get_picked_points()
                rgbd_picked = picked_points_rgbd.get_picked_points()
                n_matched_points = min(len(mesh_picked), len(rgbd_picked))

                # Registration needs a minimum of 3 points
                if n_matched_points < 3:
                    continue
                mesh_picked = mesh_picked[:n_matched_points]
                rgbd_picked = rgbd_picked[:n_matched_points]

                # Compute registration
                current_transform = p2p_registration.register(mesh_picked, rgbd_picked)

                # Update alignment visualizer
                reset_pcd(mesh_pcd_transformed, mesh_pcd)
                mesh_pcd_transformed.transform(current_transform)
                alignment_vis.update_geometry(mesh_pcd_transformed)
                print_transform(current_transform, title="Point-to-Point Result")

            # Check for ICP registration request and run ICP registration
            elif alignment_vis.icp_requested:
                # Run ICP registration
                alignment_vis.icp_requested = False
                icp_transform = icp_registration.register(
                    mesh_pcd_transformed,  # using transformed mesh to handle scale
                    rgbd_pcd,
                    initial_transform=np.eye(4),
                )
                mesh_pcd_transformed.transform(
                    icp_transform
                )  # update existing transform
                alignment_vis.update_geometry(mesh_pcd_transformed)
                current_transform = icp_transform @ current_transform
                print_transform(current_transform, title="ICP Result")

            # Limit updates to 60 Hz
            rate.sleep()

        # Save transformation
        if np.linalg.norm(current_transform - loaded_transform) < 1e-6:
            print("Transform is unchanged; not saving.")
        else:
            rotation, translation, scale = decompose_transform(current_transform)
            instance_dict.update(
                {
                    "obj_id": args.object_id,
                    "cam_t_m2c": translation,
                    "cam_R_m2c": rotation,
                    "__estimated_scale": scale,
                    "__mesh_picked_points": mesh_picked.tolist(),
                    "__rgbd_picked_points": rgbd_picked.tolist(),
                }
            )
            bop_toolkit_lib.inout.save_scene_gt(scene_gt_fn, scene_gt)
            print(f"Saved transformation to {scene_gt_fn}")

    # Close visualizers
    finally:
        alignment_vis.close()
        if not args.view_only:
            mesh_point_picker.close()
            rgbd_point_picker.close()


if __name__ == "__main__":
    main()
