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
