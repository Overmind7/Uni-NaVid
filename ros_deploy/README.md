# Uni-NaVid API Server

This folder provides a lightweight FastAPI server that wraps the `UniNaVid_Agent` for single-frame navigation inference. The server loads the model once and reuses it across requests to keep latency predictable.

## Run the server

```bash
python ros_deploy/api_server.py --host 0.0.0.0 --port 8000 --model-path /path/to/uninavid/checkpoint
```

`--model-path` overrides the `UNINAVID_MODEL_PATH` environment variable. If neither is supplied, the server defaults to `model_zoo/uninavid-7b-full-224-video-fps-1-grid-2`.

## Endpoints

### `GET /health`
Returns `{ "status": "ok" }` once the service is reachable.

### `POST /warmup`
Loads the Uni-NaVid checkpoint into memory if it is not already initialized. Response shape:

```json
{
  "status": "ready",
  "model_path": "<resolved checkpoint path>"
}
```

### `POST /predict`
Accepts one RGB frame and an instruction, and returns the action plan predicted by the model. At least one of `image_base64` or `rgb` must be supplied.

**Request body**
```json
{
  "instruction": "move to the chair and stop",
  "image_base64": "<JPEG/PNG image encoded with base64>",
  "rgb": [[[123, 231, 111], [125, 232, 112], ...]]  // optional HxWx3 array of uint8 RGB pixels
}
```

**Response body**
```json
{
  "actions": ["forward", "left", "stop"],
  "step": 1,
  "path": [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, -0.523], [0.5, 0.0, -0.523]]]
}
```

`actions` is the list of navigation tokens produced by the model. `path` mirrors the waypoints returned by `UniNaVid_Agent.act`, and `step` is the action sequence index maintained by the agent.
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
