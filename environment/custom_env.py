"""Custom Gymnasium environment for menstrual pad distribution.

The agent selects one school per step to receive delivery from a central depot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SchoolConfig:
    """Static school properties."""

    name: str
    district: str
    x: float
    y: float
    vulnerability: float
    weekly_demand: float
    capacity: float
    distance_km: float


class KigaliPadDistributionEnv(gym.Env[np.ndarray, int]):
    """RL environment for prioritizing sanitary pad deliveries across schools."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        depot_capacity: int = 500,
        max_steps: int = 28,
        low_stock_ratio_max: float = 0.30,
        delivery_per_action: int = 40,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.depot_capacity = float(depot_capacity)
        self.max_steps = int(max_steps)
        self.low_stock_ratio_max = float(low_stock_ratio_max)
        self.delivery_per_action = float(delivery_per_action)

        self.schools = self._build_school_configs()
        self.num_schools = len(self.schools)
        self.max_distance_km = max(s.distance_km for s in self.schools)
        self.max_weekly_demand = max(s.weekly_demand for s in self.schools)

        # One action per school.
        self.action_space = spaces.Discrete(self.num_schools)

        # Per school: [stock_level, weekly_demand, distance, vulnerability], flattened.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_schools * 4,),
            dtype=np.float32,
        )

        # Episode state
        self.current_step = 0
        self.current_week = 1
        self.depot_stock = self.depot_capacity
        self.school_stock = np.zeros(self.num_schools, dtype=np.float32)
        self.high_vulnerability_indices = np.array(
            [i for i, s in enumerate(self.schools) if s.vulnerability > 0.7],
            dtype=np.int32,
        )
        self.covered_high_vuln: set[int] = set()

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    @staticmethod
    def _build_school_configs() -> list[SchoolConfig]:
        """Create 12 schools inspired by Kigali districts."""
        return [
            SchoolConfig("GS Kimironko", "Gasabo", 0.82, 0.65, 0.45, 85, 160, 3.8),
            SchoolConfig("GS Kacyiru", "Gasabo", 0.74, 0.88, 0.36, 70, 150, 4.1),
            SchoolConfig("GS Gisozi", "Gasabo", 0.58, 0.79, 0.52, 80, 170, 5.2),
            SchoolConfig("GS Bumbogo", "Gasabo", 0.92, 0.95, 0.81, 95, 180, 8.4),
            SchoolConfig("GS Nyamirambo", "Nyarugenge", 0.24, 0.41, 0.57, 88, 175, 3.5),
            SchoolConfig("GS Kigali", "Nyarugenge", 0.35, 0.53, 0.40, 72, 150, 2.9),
            SchoolConfig("GS Rwezamenyo", "Nyarugenge", 0.29, 0.61, 0.49, 76, 160, 3.2),
            SchoolConfig("GS Mageragere", "Nyarugenge", 0.11, 0.27, 0.79, 90, 180, 7.8),
            SchoolConfig("GS Kanombe", "Kicukiro", 0.63, 0.24, 0.55, 84, 170, 4.9),
            SchoolConfig("GS Gikondo", "Kicukiro", 0.46, 0.22, 0.61, 86, 175, 4.2),
            SchoolConfig("GS Niboye", "Kicukiro", 0.51, 0.12, 0.68, 79, 165, 5.1),
            SchoolConfig("GS Nyarugunga", "Kicukiro", 0.77, 0.08, 0.84, 98, 190, 9.1),
        ]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.current_step = 0
        self.current_week = 1
        self.depot_stock = self.depot_capacity
        self.covered_high_vuln = set()

        capacities = np.array([s.capacity for s in self.schools], dtype=np.float32)
        self.school_stock = self.np_random.uniform(
            low=0.0,
            high=self.low_stock_ratio_max,
            size=self.num_schools,
        ).astype(np.float32) * capacities

        obs = self._get_observation()
        info = self._get_info(last_action=None, delivered=0.0)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.num_schools - 1}]")

        self.current_step += 1
        self.current_week = min(4, (self.current_step // 7) + 1)

        school = self.schools[action]
        current_stock = float(self.school_stock[action])
        stock_ratio = current_stock / school.capacity
        distance_norm = school.distance_km / self.max_distance_km

        # Delivery amount is constrained by depot stock and school remaining capacity.
        deliverable = min(
            self.delivery_per_action,
            self.depot_stock,
            school.capacity - current_stock,
        )
        delivered = max(0.0, deliverable)

        if delivered > 0.0:
            self.school_stock[action] += delivered
            self.depot_stock -= delivered
            if school.vulnerability > 0.7:
                self.covered_high_vuln.add(action)

        # Simulate one day of consumption at all schools.
        daily_demand = np.array([s.weekly_demand / 7.0 for s in self.schools], dtype=np.float32)
        self.school_stock = np.maximum(0.0, self.school_stock - daily_demand)

        reward = 0.0
        if delivered > 0.0:
            reward += 5.0
            if stock_ratio < 0.20:
                reward += 10.0

        if stock_ratio > 0.80:
            reward -= 5.0

        reward -= 2.0 * distance_norm
        reward -= 1.0  # per-step efficiency penalty

        terminated = self._is_terminal()
        truncated = self.current_step >= self.max_steps

        if (terminated or truncated) and self._all_high_vulnerability_covered():
            reward += 20.0

        obs = self._get_observation()
        info = self._get_info(last_action=action, delivered=delivered)
        return obs, float(reward), terminated, truncated, info

    def _is_terminal(self) -> bool:
        if self.depot_stock <= 0.0:
            return True
        return bool(np.all(self._stock_ratios() >= 0.60))

    def _stock_ratios(self) -> np.ndarray:
        capacities = np.array([s.capacity for s in self.schools], dtype=np.float32)
        return self.school_stock / capacities

    def _all_high_vulnerability_covered(self) -> bool:
        if len(self.high_vulnerability_indices) == 0:
            return True
        return set(self.high_vulnerability_indices.tolist()).issubset(self.covered_high_vuln)

    def _get_observation(self) -> np.ndarray:
        stock_norm = self._stock_ratios()
        demand_norm = np.array(
            [s.weekly_demand / self.max_weekly_demand for s in self.schools],
            dtype=np.float32,
        )
        distance_norm = np.array(
            [s.distance_km / self.max_distance_km for s in self.schools],
            dtype=np.float32,
        )
        vulnerability = np.array([s.vulnerability for s in self.schools], dtype=np.float32)

        features = np.stack(
            [stock_norm, demand_norm, distance_norm, vulnerability],
            axis=1,
        )
        return features.reshape(-1).astype(np.float32)

    def _get_info(self, last_action: Optional[int], delivered: float) -> dict[str, Any]:
        return {
            "step": self.current_step,
            "week": self.current_week,
            "depot_stock": float(self.depot_stock),
            "last_action": last_action,
            "last_delivery": float(delivered),
            "high_vuln_coverage": len(self.covered_high_vuln),
            "high_vuln_total": int(len(self.high_vulnerability_indices)),
        }

    def render(self) -> None:
        if self.render_mode != "human":
            return
        print(
            f"Step={self.current_step:02d} Week={self.current_week} Depot={self.depot_stock:.1f}"
        )

    def close(self) -> None:
        return
