"""FastAPI bridge between RL environment state and Three.js frontend.

Suitable for local runs and Kaggle notebooks (uvicorn in a background thread).
Shared state is protected by threading.Lock for concurrent API + agent loops.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from environment.custom_env import KigaliPadDistributionEnv


@dataclass
class VehicleState:
    x: float
    y: float
    target_school_id: Optional[int]
    moving: bool


class VisualizationBridge:
    """Holds shared simulation state; all public reads/writes use the lock."""

    def __init__(self, env: Optional[KigaliPadDistributionEnv] = None) -> None:
        self.lock = threading.Lock()
        self.env = env or KigaliPadDistributionEnv()
        self.obs, _ = self.env.reset(seed=42)
        self.total_reward = 0.0
        self.done = False
        self.last_action: Optional[int] = None
        self.vehicle = VehicleState(x=0.5, y=0.5, target_school_id=None, moving=False)
        self.current_episode = 1
        self._model: Any = None
        self.last_step_reward: float = 0.0

    def set_model(self, model: Any) -> None:
        """Attach a stable-baselines3 model (e.g. DQN) for policy actions."""
        with self.lock:
            self._model = model

    def _school_payload(self, school_id: int) -> dict[str, Any]:
        school = self.env.schools[school_id]
        ratio = float(self.env._stock_ratios()[school_id])
        return {
            "id": school_id,
            "name": school.name,
            "district": school.district,
            "x": float(school.x),
            "y": float(school.y),
            "stock_ratio": ratio,
            "vulnerability": float(school.vulnerability),
            "weekly_demand": float(school.weekly_demand),
            "is_high_vulnerability": bool(school.vulnerability > 0.7),
        }

    def state_payload(self) -> dict[str, Any]:
        """Build JSON snapshot. Call only while holding ``self.lock``."""
        schools = [self._school_payload(i) for i in range(self.env.num_schools)]
        last_action_name = (
            self.env.schools[self.last_action].name if self.last_action is not None else None
        )
        return {
            "episode": self.current_episode,
            "step": int(self.env.current_step),
            "week": int(self.env.current_week),
            "total_reward": float(self.total_reward),
            "depot_stock": float(self.env.depot_stock),
            "last_action": self.last_action,
            "last_action_name": last_action_name,
            "done": bool(self.done),
            "vehicle": {
                "x": float(self.vehicle.x),
                "y": float(self.vehicle.y),
                "target_school_id": self.vehicle.target_school_id,
                "moving": self.vehicle.moving,
            },
            "schools": schools,
            "last_step_reward": float(self.last_step_reward),
        }

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return self.state_payload()

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        with self.lock:
            self.obs, _ = self.env.reset(seed=seed)
            self.total_reward = 0.0
            self.done = False
            self.last_action = None
            self.vehicle = VehicleState(x=0.5, y=0.5, target_school_id=None, moving=False)
            self.current_episode += 1
            self.last_step_reward = 0.0
            return self.state_payload()

    def _predict_action(self) -> int:
        if self._model is None:
            stock = self.env._stock_ratios()
            vuln = np.array([s.vulnerability for s in self.env.schools], dtype=np.float32)
            dist = np.array([s.distance_km for s in self.env.schools], dtype=np.float32)
            score = (1.0 - stock) * 0.65 + vuln * 0.30 - (dist / self.env.max_distance_km) * 0.05
            return int(np.argmax(score))
        action, _ = self._model.predict(self.obs, deterministic=True)
        return int(np.asarray(action).item())

    def step_with_model(self) -> dict[str, Any]:
        """One env step using the loaded model (or heuristic if no model)."""
        with self.lock:
            if self.done:
                return self.state_payload()
            action = self._predict_action()
            if not self.env.action_space.contains(action):
                raise ValueError(f"Invalid action from model: {action}")
            self.obs, reward, terminated, truncated, _ = self.env.step(action)
            self.last_step_reward = float(reward)
            self.total_reward += float(reward)
            self.done = bool(terminated or truncated)
            self.last_action = int(action)
            target = self.env.schools[action]
            self.vehicle = VehicleState(
                x=float(target.x),
                y=float(target.y),
                target_school_id=action,
                moving=True,
            )
            return self.state_payload()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_app(bridge: Optional[VisualizationBridge] = None) -> FastAPI:
    """Build FastAPI app. Pass a preconfigured ``bridge`` (e.g. with DQN) for Kaggle."""
    app = FastAPI(title="Kigali RL Visualization API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if bridge is None:
        bridge = VisualizationBridge()
    visual_dir = _repo_root() / "visualization"
    visual_resolved = visual_dir.resolve()

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(str(visual_dir / "index.html"))

    @app.get("/static/{filename}")
    def serve_static(filename: str) -> FileResponse:
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=404, detail="Invalid path")
        path = (visual_dir / filename).resolve()
        if not str(path).startswith(str(visual_resolved)):
            raise HTTPException(status_code=404, detail="Invalid path")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(str(path))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/state")
    def state() -> dict[str, Any]:
        return bridge.get_state()

    @app.post("/step")
    def step() -> dict[str, Any]:
        try:
            return bridge.step_with_model()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/reset")
    def reset(seed: Optional[int] = Query(default=None)) -> dict[str, Any]:
        return bridge.reset(seed=seed)

    return app


# Default app for: uvicorn environment.rendering:app
app = create_app()
