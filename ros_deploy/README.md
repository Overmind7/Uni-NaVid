# ROS Deployment for Uni-NaVid

This package wraps the `UniNaVid_Agent` in a ROS node that consumes RGB camera frames and publishes velocity commands via `geometry_msgs/Twist`.

## Contents
- `ros_deploy/node.py`: ROS node that subscribes to RGB frames and camera info, runs `UniNaVid_Agent.act`, and publishes velocity commands.
- `ros_deploy/srv/SetInstruction.srv`: Service to update the navigation instruction string at runtime.
- `config/default.yaml`: Default topic names, model path, and velocity scaling values.

## Configuration
Update `ros_deploy/config/default.yaml` or supply a different file via the `~config_path` parameter. Example:

```yaml
image_topic: "/rscamera_front/color/image"
camera_info_topic: "/rscamera_front/color/camera_info"
cmd_vel_topic: "/cmd_vel"
model_path: "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
linear_speed: 0.25
angular_speed: 0.5
default_instruction: "Please navigate according to the current mission."
```

## Building

```bash
cd /path/to/catkin_ws/src
git clone https://github.com/your-org/Uni-NaVid.git
cd ..
catkin_make
source devel/setup.bash
```

## Running the node

Start the node after sourcing your workspace:

```bash
rosrun ros_deploy node.py _config_path:=/full/path/to/config.yaml _instruction:="Find the kitchen"
```

If you rely on the default configuration, simply run:

```bash
rosrun ros_deploy node.py
```

## Updating the instruction

- **Service call**:

```bash
rosservice call /uninavid_agent/set_instruction "instruction: 'Turn left at the hallway'"
```

- **Parameter update**:

```bash
rosparam set /uninavid_agent/instruction "Turn right at the next door"
```

The node will use the latest instruction when processing incoming frames.
