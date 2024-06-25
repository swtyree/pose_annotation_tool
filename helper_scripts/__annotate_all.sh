#!/bin/bash

# get capture path and object/instance ids from args
SOURCE_PATH=realpath $(dirname $0)/..
MESHES_PATH_ANNOT=../meshes/release_fixed_scale/meshes_annot
MESHES_PATH_RENDER=../meshes/release_fixed_scale/meshes_ply
SCENE_NAME=$1
SCENE_PATH=$SCENE_NAME/bop_annot

# initialize `scene_gt_raw.json`
if [ ! -f $SCENE_PATH/scene_gt_raw.json ]; then
    echo "Initializing scene_gt_raw.json..."
    cp $SCENE_PATH/scene_gt_empty.json $SCENE_PATH/scene_gt_raw.json
fi

# get list of object IDs
OBJ_ID_FN=../scene_objects.json
OBJ_IDS=$(cat $OBJ_ID_FN | jq ".$SCENE_NAME.bop_objects" | jq -r ".[]")

# iterate over object IDs
for OBJ_ID in $OBJ_IDS; do
    echo "Annotating object $OBJ_ID in $SCENE_NAME..."

    # check for existing annotation in `sceen_gt_empty.json`
    # TODO
    
    # iterate over view IDs in `scene_camera_raw.json`
    VIEW_ID_FN=$SCENE_PATH/scene_camera_raw.json
    for VIEW_ID in $(jq 'keys[]' $VIEW_ID_FN | jq -r 'tonumber' | sort -n); do
        echo "Loading view $VIEW_ID..."
        python $SOURCE_PATH/main.py \
            --meshes_dir $MESHES_PATH_ANNOT \
            --scene_dir $SCENE_PATH \
            --view_id $VIEW_ID \
            --instance_id -1 \
            --object_id $OBJ_ID \
            --rgbd_cache_dir $SCENE_PATH/rgbd_cache \
            --scene_gt_fn scene_gt_raw.json \
            --crop && break  # we found a good view and successfully annotated it
            # --optimize_scale
    done
    echo

done
echo

# propagate poses to all views
python $SOURCE_PATH/propagate_poses.py \
    scene_gt_raw.json \
    scene_camera.json \
    scene_gt.json \
    --overwrite
echo "Poses propagated to all views in $SCENE_NAME/scene_gt.json"

# render reprojections
python $SOURCE_PATH/vis_poses.py 
    $MESHES_PATH_RENDER \
    $OBJ_IDS \
    $SCENE_PATH/scene_camera.json \
    $SCENE_PATH/scene_gt.json \
    $SCENE_PATH/rgb/ \
    $SCENE_PATH/vis/
echo "Visualizations rendered to $SCENE_PATH/vis/"
echo "Done!"
