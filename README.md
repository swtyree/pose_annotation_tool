Typical usage:

```
python ~/code/pose_annotation_tool/main.py \
    --meshes_dir $MESHES_PATH \
    --scene_dir $SCENE_PATH \
    --view_id $VIEW_ID \
    --instance_id -1 \
    --object_id $OBJ_ID \
    --rgbd_cache_dir $SCENE_PATH/rgbd_cache \
    --optimize_scale \
    --crop
```