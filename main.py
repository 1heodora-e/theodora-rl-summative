"""Entry point for FastAPI + Three.js visualization."""

from __future__ import annotations

import uvicorn

from environment.rendering import create_app


def run() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    run()

