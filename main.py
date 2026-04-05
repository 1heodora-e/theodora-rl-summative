"""Entry point: FastAPI + Three.js visualization with the best DQN policy (local default)."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from stable_baselines3 import DQN

from environment.custom_env import KigaliPadDistributionEnv
from environment.rendering import VisualizationBridge, create_app


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def run() -> None:
    model_path = _repo_root() / "models" / "dqn" / "dqn_run_09.zip"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"DQN checkpoint not found: {model_path}\n"
            "Place models/dqn/dqn_run_09.zip in the repo or train via notebooks/model-training.ipynb."
        )
    bridge = VisualizationBridge(KigaliPadDistributionEnv())
    bridge.set_model(DQN.load(str(model_path)))
    app = create_app(bridge)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    run()
