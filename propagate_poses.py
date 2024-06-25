import argparse
from pathlib import Path

import numpy as np
from bop_toolkit_lib import inout

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("input_scene_gt_fn", type=Path, help="Input scene GT file")
parser.add_argument("input_scene_camera_fn", type=Path, help="Input scene camera file")
parser.add_argument("output_scene_gt_fn", type=Path, help="Output scene GT file")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Allow existing output files to be overwritten",
)
args = parser.parse_args()

# validate inputs
assert args.input_scene_gt_fn.exists(), f"{args.input_scene_gt_fn} does not exist"
assert (
    args.input_scene_camera_fn.exists()
), f"{args.input_scene_camera_fn} does not exist"
assert (
    args.overwrite or not args.output_scene_gt_fn.exists()
), f"{args.output_scene_gt_fn} already exists; use --overwrite"

# read scene_gt.json and scene_camera.json
with open(args.input_scene_gt_fn, "r") as f:
    scene_gt = inout.load_scene_gt(args.input_scene_gt_fn)
with open(args.input_scene_camera_fn, "r") as f:
    scene_camera = inout.load_scene_camera(args.input_scene_camera_fn)

# get object poses (m2c) for all objects, wrt the first camera
objects = []
for view_idx, view in scene_gt.items():
    # get view camera
    camera_w2c = np.eye(4)
    camera_w2c[:3, 3] = scene_camera[view_idx]["cam_t_w2c"].flatten()
    camera_w2c[:3, :3] = scene_camera[view_idx]["cam_R_w2c"]

    # iterate over object poses in view
    for pose in view:
        # get object pose
        obj_m2c = np.eye(4)
        obj_m2c[:3, 3] = pose["cam_t_m2c"].flatten() / 1000  # TODO why?!?!?!
        obj_m2c[:3, :3] = pose["cam_R_m2c"]

        # get object pose wrt the first camera
        obj_m2w = np.linalg.inv(camera_w2c) @ obj_m2c
        objects.append(
            {
                "obj_id": pose["obj_id"],
                "obj_m2w": obj_m2w,
            }
        )

# propagate object poses to all cameras
scene_gt_prop = {}
for view_id, camera in scene_camera.items():
    # get camera pose (w2c)
    camera_w2c = np.eye(4)
    camera_w2c[:3, 3] = camera["cam_t_w2c"].flatten()
    camera_w2c[:3, :3] = camera["cam_R_w2c"]

    # get object poses for this camera
    scene_gt_prop[view_id] = []
    for obj in objects:
        # get object pose wrt this camera
        obj_m2c = camera_w2c @ obj["obj_m2w"]
        scene_gt_prop[view_id].append(
            {
                "obj_id": obj["obj_id"],
                "cam_t_m2c": obj_m2c[:3, 3].flatten(),
                "cam_R_m2c": obj_m2c[:3, :3],
            }
        )

# save scene_gt_prop.json
inout.save_scene_gt(args.output_scene_gt_fn, scene_gt_prop)
print(f"Saved {args.output_scene_gt_fn}")
