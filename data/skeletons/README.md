# Skeletons

These files correspond to different 'skeletons' that models can use.

Each skeleton specifies the detectable joints on the robot.

As they are taken from DeepPoseKit, some functionality is unused, mainly the *swap* column, which can be disregarded.

Skeletons are of the form:

| name         | parent       | swap  |
| ------------ | ------------ | ----- |
| joint_1_name |              |       |
| joint_2_name | joint_1_name |       |
| joint_3_name | joint_2_name |       |

# Keypoint Configuration

Each Skeleton should have a corresposing keypoint configuration JSON if you wish to use automatic annotation.

The following template is provided:
```json
{
    "markers":{
        "height": 0.05,
        "radius": 0.05
    },
    "keypoints":{
        "keypoint":{
            "parent_joint": "A Joint Name",
            "pose":[1,1,1,0,0,0]
        },
        "another_keypoint":{
            "parent_joint": "Another Joint Name",
            "pose":[0,0,0,1,1,1]
        }
    }
}

```

Each keypoint should have a distinct name.

Keypoint parent joints should be specified by the same name that will be given to the joint meshes in rendering.

A general rule of thumb for keypoint poses is the following:
```json
{
"pose":[x,0,0,1.5707963267948966,0,0]
}
```
Where x is the distance from the mesh midpoint to the surface of the mesh.