"""FastAPI server for Uni-NaVid navigation inference.

This server exposes endpoints to run Uni-NaVid on single RGB frames.
"""
import argparse
import base64
import os
import sys
import threading
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# Ensure repository root is on sys.path even when executed from inside the
# ros_deploy package directory (e.g., `cd ros_deploy && python api_server.py`).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from offline_eval_uninavid import UniNaVid_Agent

DEFAULT_MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"

app = FastAPI(title="Uni-NaVid API", version="1.0")

_agent: Optional[UniNaVid_Agent] = None
_agent_lock = threading.Lock()


class InferenceRequest(BaseModel):
    """Schema for navigation inference."""

    instruction: str
    image_base64: Optional[str] = None
    rgb: Optional[List[List[List[int]]]] = None


class InferenceResponse(BaseModel):
    """Schema for returned actions."""

    actions: List[str]
    step: int
    path: List[List[List[float]]]


class WarmupResponse(BaseModel):
    status: str
    model_path: str


def _get_model_path() -> str:
    return os.getenv("UNINAVID_MODEL_PATH", DEFAULT_MODEL_PATH)


def get_agent() -> UniNaVid_Agent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                model_path = _get_model_path()
                _agent = UniNaVid_Agent(model_path)
    return _agent


def _decode_image(request: InferenceRequest) -> np.ndarray:
    if request.image_base64:
        try:
            binary = base64.b64decode(request.image_base64)
            image_array = np.frombuffer(binary, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc
        if frame is None:
            raise HTTPException(status_code=400, detail="Unable to decode base64 image")
        return frame

    if request.rgb is not None:
        frame = np.asarray(request.rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise HTTPException(status_code=400, detail="RGB array must have shape [H, W, 3]")
        return frame

    raise HTTPException(status_code=400, detail="Provide either image_base64 or rgb in the request body")


@app.get("/health", summary="Service health probe")
def health() -> dict:
    return {"status": "ok"}


@app.post("/warmup", response_model=WarmupResponse, summary="Load model into memory")
def warmup() -> WarmupResponse:
    agent = get_agent()
    return WarmupResponse(status="ready", model_path=_get_model_path())


@app.post("/predict", response_model=InferenceResponse, summary="Get navigation actions")
def predict(request: InferenceRequest) -> InferenceResponse:
    frame = _decode_image(request)
    agent = get_agent()
    try:
        result = agent.act({"observations": frame, "instruction": request.instruction})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return InferenceResponse(
        actions=result.get("actions", []),
        step=result.get("step", 0),
        path=result.get("path", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Uni-NaVid FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind")
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Path to Uni-NaVid checkpoint (defaults to env UNINAVID_MODEL_PATH or {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    if args.model_path:
        os.environ["UNINAVID_MODEL_PATH"] = args.model_path

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
