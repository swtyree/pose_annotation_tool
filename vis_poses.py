# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in the ground-truth poses."""

import argparse
import os
from pathlib import Path

import numpy as np
import tqdm
from bop_toolkit_lib import config, inout, renderer, visualization

# Parse arguments
parser = argparse.ArgumentParser(
    description="Visualize object models in the ground-truth poses"
)
parser.add_argument("mesh_path", type=Path, help="Path to mesh directory")
# parser.add_argument("input_mesh_fn", type=Path, help="Input mesh file")
parser.add_argument("bop_obj_ids", type=int, nargs="+", help="BOP object ID")
parser.add_argument("scene_camera_fn", type=Path, help="Scene camera file")
parser.add_argument("scene_gt_fn", type=Path, help="Scene GT file")
parser.add_argument("rgb_path", type=Path, help="Path to RGB images")
parser.add_argument("output_path", type=Path, help="Path to output RGB images")
parser.add_argument(
    "--rgb_tpath", type=str, default="%06d.jpg", help="Template for RGB image paths"
)
parser.add_argument("--png", action="store_true", help="RGB images are in PNG format")
parser.add_argument("--skip", type=int, help="Skip images", default=1)
parser.add_argument("--start", type=int, help="Start image", default=0)
args = parser.parse_args()

if args.png:
    args.rgb_tpath = args.rgb_tpath.replace(".jpg", ".png")

# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "lm",
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "test",
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # File with a list of estimation targets used to determine the set of images
    # for which the GT poses will be visualized. The file is assumed to be stored
    # in the dataset folder. None = all images.
    # 'targets_filename': 'test_targets_bop19.json',
    "targets_filename": None,
    # Select ID's of scenes, images and GT poses to be processed.
    # Empty list [] means that all ID's will be used.
    "scene_ids": [],
    "im_ids": [],
    "gt_ids": [],
    # Indicates whether to render RGB images.
    "vis_rgb": True,
    # Indicates whether to resolve visibility in the rendered RGB images (using
    # depth renderings). If True, only the part of object surface, which is not
    # occluded by any other modeled object, is visible. If False, RGB renderings
    # of individual objects are blended together.
    "vis_rgb_resolve_visib": False,
    # Indicates whether to save images of depth differences.
    "vis_depth_diff": False,
    # Whether to use the original model color.
    "vis_orig_color": False,
    # Type of the renderer (used for the VSD pose error function).
    "renderer_type": "vispy",  # Options: 'vispy', 'cpp', 'python'.
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
    # Folder for output visualisations.
    "vis_path": os.path.join(config.output_path, "vis_gt_poses"),
    # Path templates for output images.
    "vis_rgb_tpath": os.path.join(
        "{vis_path}", "{dataset}", "{split}", "{scene_id:06d}", "{im_id:06d}.jpg"
    ),
    "vis_depth_diff_tpath": os.path.join(
        "{vis_path}",
        "{dataset}",
        "{split}",
        "{scene_id:06d}",
        "{im_id:06d}_depth_diff.jpg",
    ),
}
################################################################################

# Load scene info and ground-truth poses.
scene_camera = inout.load_scene_camera(args.scene_camera_fn.as_posix())
scene_gt = inout.load_scene_gt(args.scene_gt_fn.as_posix())

# # Load dataset parameters.
# dp_split = dataset_params.get_split_params(
#     p["datasets_path"], p["dataset"], p["dataset_split"], p["dataset_split_type"]
# )

# model_type = "eval"  # None = default.
# dp_model = dataset_params.get_model_params(p["datasets_path"], p["dataset"], model_type)

# Load colors.
# colors_path = os.path.join(os.path.dirname(visualization.__file__), "colors.json")
# colors = inout.load_json(colors_path)

# Subset of images for which the ground-truth poses will be rendered.
# if p["targets_filename"] is not None:
#     targets = inout.load_json(
#         os.path.join(dp_split["base_path"], p["targets_filename"])
#     )
#     scene_im_ids = {}
#     for target in targets:
#         scene_im_ids.setdefault(target["scene_id"], set()).add(target["im_id"])
# else:
#     scene_im_ids = None

# List of considered scenes.
# scene_ids_curr = dp_split["scene_ids"]
# if p["scene_ids"]:
#     scene_ids_curr = set(scene_ids_curr).intersection(p["scene_ids"])

# Rendering mode.
# renderer_modalities = []
# if p["vis_rgb"]:
#     renderer_modalities.append("rgb")
# if p["vis_depth_diff"] or (p["vis_rgb"] and p["vis_rgb_resolve_visib"]):
#     renderer_modalities.append("depth")
# renderer_mode = "+".join(renderer_modalities)
renderer_modalities = ["rgb"]
renderer_mode = "+".join(renderer_modalities)

# Create a renderer.
first_view_idx, camera_first_view = list(scene_camera.items())[0]
if "width" in camera_first_view and "height" in camera_first_view:
    width, height = camera_first_view["width"], camera_first_view["height"]
else:
    first_rgb_fn = args.rgb_path / (args.rgb_tpath % first_view_idx)
    first_rgb = inout.load_im(first_rgb_fn)[:, :, :3]
    width, height = first_rgb.shape[1], first_rgb.shape[0]
ren = renderer.create_renderer(
    width, height, p["renderer_type"], mode=renderer_mode, shading="flat"
)

# Load object models.
print("Loading 3D model of object(s)...")
for obj_id in args.bop_obj_ids:
    model_path = args.mesh_path / f"obj_{obj_id:06d}.ply"
    model_color = [0.89, 0.28, 0.13]
    # ren.add_object(obj_id, model_path, surf_color=model_color)
    ren.add_object(obj_id, model_path)

# List of considered images.
im_ids = sorted(scene_gt.keys())
if args.start:
    im_ids = im_ids[args.start :]
if args.skip:
    im_ids = im_ids[:: args.skip]

# Render the object models in the ground-truth poses in the selected images.
for im_id in tqdm.tqdm(im_ids):
    K = scene_camera[im_id]["cam_K"]

    # List of considered ground-truth poses.
    gt_ids_curr = range(len(scene_gt[im_id]))
    if p["gt_ids"]:
        gt_ids_curr = set(gt_ids_curr).intersection(p["gt_ids"])

    # Collect the ground-truth poses.
    gt_poses = []
    for gt_id in gt_ids_curr:
        gt = scene_gt[im_id][gt_id]
        gt_poses.append(
            {
                "obj_id": gt["obj_id"],
                "R": gt["cam_R_m2c"],
                "t": gt["cam_t_m2c"],
                # "text_info": [
                #     {
                #         "name": "",
                #         "val": "{}:{}".format(gt["obj_id"], gt_id),
                #         "fmt": "",
                #     }
                # ],
            }
        )

    # Load the color and depth images and prepare images for rendering.
    rgb = inout.load_im(args.rgb_path / (args.rgb_tpath % im_id))[:, :, :3]
    depth = None
    # if p["vis_depth_diff"] or (p["vis_rgb"] and p["vis_rgb_resolve_visib"]):
    #     depth = inout.load_depth(
    #         dp_split["depth_tpath"].format(scene_id=scene_id, im_id=im_id)
    #     )
    #     depth *= scene_camera[im_id]["depth_scale"]  # Convert to [mm].

    # Path to the output RGB visualization.
    # vis_rgb_path = None
    # if p["vis_rgb"]:
    #     vis_rgb_path = p["vis_rgb_tpath"].format(
    #         vis_path=p["vis_path"],
    #         dataset=p["dataset"],
    #         split=p["dataset_split"],
    #         scene_id=scene_id,
    #         im_id=im_id,
    #     )
    args.output_path.mkdir(parents=True, exist_ok=True)
    vis_rgb_path = (args.output_path / (args.rgb_tpath % im_id)).as_posix()

    # Path to the output depth difference visualization.
    # vis_depth_diff_path = None
    # if p["vis_depth_diff"]:
    #     vis_depth_diff_path = p["vis_depth_diff_tpath"].format(
    #         vis_path=p["vis_path"],
    #         dataset=p["dataset"],
    #         split=p["dataset_split"],
    #         scene_id=scene_id,
    #         im_id=im_id,
    #     )

    # Visualization.
    visualization.vis_object_poses(
        poses=gt_poses,
        K=K,
        renderer=ren,
        rgb=rgb,
        depth=depth,
        vis_rgb_path=vis_rgb_path,
        vis_depth_diff_path=None,
        vis_rgb_resolve_visib=False,
    )

print("Done.")
