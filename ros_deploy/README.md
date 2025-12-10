# Uni-NaVid API 服务

该目录提供一个轻量级的 FastAPI 服务器，将 `UniNaVid_Agent` 封装为单帧导航推理服务。服务器启动时只加载一次模型并在请求之间重复复用，保持可预期的延迟。

## 启动服务器

```bash
python ros_deploy/api_server.py --host 0.0.0.0 --port 8000 --model-path /path/to/uninavid/checkpoint
```

`--model-path` 可覆盖环境变量 `UNINAVID_MODEL_PATH`。如果两者都未提供，服务器默认使用 `model_zoo/uninavid-7b-full-224-video-fps-1-grid-2`。

## 端点

### `GET /health`
服务可达时返回 `{ "status": "ok" }`。

### `POST /warmup`
若尚未初始化，将 Uni-NaVid 权重加载到内存。响应格式如下：

```json
{
  "status": "ready",
  "model_path": "<resolved checkpoint path>"
}
```

### `POST /predict`
接受一张 RGB 图片和一段指令，返回模型预测的行动规划。`image_base64` 与 `rgb` 至少需提供其一。

**请求示例**
```json
{
  "instruction": "move to the chair and stop",
  "image_base64": "<JPEG/PNG image encoded with base64>",
  "rgb": [[[123, 231, 111], [125, 232, 112], ...]]  // 可选：HxWx3 的 uint8 RGB 数组
}
```

**响应示例**
```json
{
  "actions": ["forward", "left", "stop"],
  "step": 1,
  "path": [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, -0.523], [0.5, 0.0, -0.523]]]
}
```

`actions` 为模型输出的导航 token；`path` 对应 `UniNaVid_Agent.act` 返回的路径点，`step` 则是代理维护的动作序列索引。

# Uni-NaVid 的 ROS 部署

此软件包将 `UniNaVid_Agent` 封装成 ROS 节点，订阅 RGB 相机图像并通过 `geometry_msgs/Twist` 发布速度指令。

## 目录内容
- `ros_deploy/node.py`：订阅 RGB 图像与相机信息，调用 `UniNaVid_Agent.act` 并发布速度指令的 ROS 节点。
- `ros_deploy/api_client_node.py`：调用远端 Uni-NaVid API 服务，将返回的离散动作转换为 `geometry_msgs/Twist` 的 ROS 节点。
- `ros_deploy/srv/SetInstruction.srv`：运行时更新导航指令字符串的服务接口。
- `config/default.yaml`：默认主题名称、模型路径与速度倍率配置。
- `config/api_client.yaml`：使用 API 服务时的默认主题、服务器地址与超时设置。

## 配置
修改 `ros_deploy/config/default.yaml`，或通过 `~config_path` 参数指定其他配置文件。示例如下：

```yaml
image_topic: "/rscamera_front/color/image"
camera_info_topic: "/rscamera_front/color/camera_info"
cmd_vel_topic: "/cmd_vel"
model_path: "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
linear_speed: 0.25
angular_speed: 0.5
default_instruction: "Please navigate according to the current mission."
```

## 编译

```bash
cd /path/to/catkin_ws/src
git clone https://github.com/your-org/Uni-NaVid.git
cd ..
catkin_make
source devel/setup.bash
```

## 运行节点

在已 source 工作空间后启动节点：

```bash
rosrun ros_deploy node.py _config_path:=/full/path/to/config.yaml _instruction:="Find the kitchen"
```

若使用默认配置文件，直接执行：

```bash
rosrun ros_deploy node.py
```

## 通过 API 服务运行节点

当模型部署在服务器上时，可使用 API 客户端节点将机器人采集的图像发送给服务器并发布速度指令：

```bash
rosrun ros_deploy api_client_node.py _server_url:=http://<server-ip>:8000 _instruction:="Find the kitchen"
```

如需自定义参数，可在配置文件中修改或通过参数重载：

```yaml
image_topic: "/rscamera_front/color/image"
cmd_vel_topic: "/cmd_vel"
server_url: "http://127.0.0.1:8000"
linear_speed: 0.25
angular_speed: 0.5
request_timeout: 5.0
default_instruction: "Please navigate according to the current mission."
```

运行时也可以指定配置路径：

```bash
rosrun ros_deploy api_client_node.py _config_path:=/full/path/to/api_client.yaml
```

## 更新指令

- **通过服务调用**：

```bash
rosservice call /uninavid_agent/set_instruction "instruction: 'Turn left at the hallway'"
```

- **通过参数更新**：

```bash
rosparam set /uninavid_agent/instruction "Turn right at the next door"
```

节点会在处理新图像时使用最新的指令。
