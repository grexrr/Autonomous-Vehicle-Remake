import numpy as np
import numpy.typing as npt
from typing import Any, Optional
from scipy.spatial import KDTree

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AutonomousVehicle.modeling.obstacles import Obstacles
from AutonomousVehicle.modeling.car import Car
from api.event_types import GLOBAL_PLANNER_TRAJECTORY, KNOWN_OBSTACLES_UPDATED, NEW_OBSTACLES_DISCOVERED, TRAJECTORY_COLLIDED
from .event_bus import EventBus

DISCARD_FIRST_N = 5

class TrajectoryCollisionChecker:
    def __init__(self, trajectory: npt.NDArray[np.floating[Any]]) -> None:
        assert trajectory.ndim == 2 and trajectory.shape[1] == 3, "trajectory must be 2D array having [[x, y, yaw]]"
        self._trajectory = trajectory

        # Calculate the trajectory of the center of the car, instead of the center of the rear axle
        xy, yaw = trajectory[:, :2], trajectory[:, 2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        xy = (xy.T + [Car.BACK_TO_CENTER * cy, Car.BACK_TO_CENTER * sy]).T
        self._trajectory_kd_tree = KDTree(xy)

    def check(self, obstacles: Obstacles) -> bool:
        indices = self._trajectory_kd_tree.query_ball_tree(obstacles.kd_tree, Car.COLLISION_RADIUS)
        for i, ids in enumerate(indices):
            if not ids:
                continue
            if Car(*self._trajectory[i]).check_collision(obstacles.coordinates[ids]):
                return True
        return False

class TrajectoryCollisionCheckingAdapter:
    """
    Trajectory Collision Checking Adapter
    
    This adapter:
    1. Checks if a trajectory collides with obstacles
    2. Subscribes to trajectory and obstacle update events
    3. Publishes collision events via EventBus
    
    Events subscribed:
    - GLOBAL_PLANNER_TRAJECTORY: (trajectory: np.ndarray) - New trajectory to check
    - KNOWN_OBSTACLES_UPDATED: (obstacles: np.ndarray) - Known obstacles updated
    - NEW_OBSTACLES_DISCOVERED: (obstacles: np.ndarray) - New obstacles discovered
    
    Events published:
    - TRAJECTORY_COLLIDED: () - Collision detected
    """

    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize trajectory collision checking adapter
        
        Args:
            event_bus: EventBus instance for publishing events
        """
        self._event_bus = event_bus
        self._checker: Optional[TrajectoryCollisionChecker] = None
        self._known_obstacles: Optional[npt.NDArray[np.floating[Any]]] = None

        self._event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, self._on_trajectory)
        self._event_bus.subscribe(KNOWN_OBSTACLES_UPDATED, self._on_known_obstacles_updated)
        self._event_bus.subscribe(NEW_OBSTACLES_DISCOVERED, self._on_new_obstacles_discovered)

    
    def _on_trajectory(self, trajectory: Optional[npt.NDArray[np.floating[Any]]]) -> None:
        """
        Handle new trajectory from global planner
        
        Args:
            trajectory: Global trajectory or None
        """
        if trajectory is None:
            self._checker = None
            return

        # 创建碰撞检测器（丢弃前N个点）
        self._checker = TrajectoryCollisionChecker(trajectory[DISCARD_FIRST_N:, :3])
        if self._known_obstacles:
            self._check_collision(self._known_obstacles)


    def _on_known_obstacles_updated(self, known_obstacles: npt.NDArray[np.floating[Any]]) -> None:
        """
        Handle known obstacles update from map server
        
        Args:
            known_obstacles: Updated known obstacle coordinates
        """
        self._known_obstacles = known_obstacles
        if self._checker:
            self._check_collision(known_obstacles)

    def _on_new_obstacles_discovered(self, new_obstacles: npt.NDArray[np.floating[Any]]) -> None:
        """
        Handle new obstacles discovered by LIDAR
        
        Args:
            new_obstacles: Newly discovered obstacle coordinates
        """
        if self._checker:
            self._check_collision(new_obstacles)
    
    def _check_collision(self, obstacle_coordinates: npt.NDArray[np.floating[Any]]) -> None:
        """
        Check if trajectory collides with obstacles
        
        Args:
            obstacle_coordinates: Obstacle coordinates to check against
        """
        if self._checker and self._checker.check(Obstacles(obstacle_coordinates)):
            self._event_bus.emit(TRAJECTORY_COLLIDED)
    
    def cancel(self) -> None:
        """Cancel collision checking (clear current trajectory)"""
        self._checker = None
    
    def stop(self) -> None:
        """Stop and clean up resources"""
        self._event_bus.unsubscribe(GLOBAL_PLANNER_TRAJECTORY, self._on_trajectory)
        self._event_bus.unsubscribe(KNOWN_OBSTACLES_UPDATED, self._on_known_obstacles_updated)
        self._event_bus.unsubscribe(NEW_OBSTACLES_DISCOVERED, self._on_new_obstacles_discovered)

        self._checker = None
        self._known_obstacles = None
        
    
