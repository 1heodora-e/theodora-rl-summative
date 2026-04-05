"""
Kaggle notebook driver: DQN visualization with FastAPI + pyngrok.

Run inside a Kaggle notebook cell, e.g.:
    !pip install -q pyngrok uvicorn fastapi stable-baselines3 gymnasium
    %run /kaggle/working/.../notebooks/kaggle_visualization.py

Or copy-paste into a cell after uploading this repo as a dataset.
Set NGROK_AUTHTOKEN in Kaggle secrets for a stable tunnel (recommended).
"""

from __future__ import annotations

import os
import sys
import threading
import time

# 1) Dataset path (adjust if your Kaggle dataset slug differs)
REPO_DIR = os.environ.get(
    "REPO_DIR",
    "/kaggle/input/datasets/theodoraegbunike/rl-summative-menstrual/theodora-rl-summative",
)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# 3) Environment import (ensures package layout is valid)
from environment.custom_env import KigaliPadDistributionEnv  # noqa: E402

# 4) FastAPI factory + bridge
from environment.rendering import VisualizationBridge, create_app  # noqa: E402

from stable_baselines3 import DQN  # noqa: E402
import uvicorn  # noqa: E402

MODEL_REL = os.path.join("models", "dqn", "dqn_run_09.zip")
MODEL_PATH = os.path.join(REPO_DIR, MODEL_REL)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"DQN model not found: {MODEL_PATH}\n"
        "Upload models/dqn/dqn_run_09.zip inside your Kaggle dataset or set REPO_DIR."
    )

# Shared bridge + env + model
bridge = VisualizationBridge(KigaliPadDistributionEnv())
bridge.set_model(DQN.load(MODEL_PATH))
app = create_app(bridge)


def _run_uvicorn() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


def _agent_loop() -> None:
    """Step the environment every 0.5s so GET /state reflects live play."""
    while True:
        time.sleep(0.5)
        if bridge.get_state()["done"]:
            bridge.reset(seed=None)
            continue
        bridge.step_with_model()


if __name__ == "__main__":
    # 6) Uvicorn in background thread
    server_thread = threading.Thread(target=_run_uvicorn, daemon=True)
    server_thread.start()
    time.sleep(2)  # let socket bind

    # 7) pyngrok public URL
    try:
        from pyngrok import ngrok
    except ImportError as e:
        raise ImportError("Install pyngrok: pip install pyngrok") from e

    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    tunnel = ngrok.connect(8080, bind_tls=True)
    public_url = tunnel.public_url if hasattr(tunnel, "public_url") else str(tunnel)
    print("Public visualization URL:", public_url)
    print("Open this URL in your browser. The scene updates from GET /state every 500ms.")

    # 8) Agent loop (steps env; API readers use the same bridge under lock)
    agent_thread = threading.Thread(target=_agent_loop, daemon=True)
    agent_thread.start()

    # Keep main thread alive in script context
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        ngrok.disconnect(public_url)
        ngrok.kill()
