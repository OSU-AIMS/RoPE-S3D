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