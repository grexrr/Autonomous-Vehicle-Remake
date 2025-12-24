from itertools import chain
from typing import Any

import cv2 as cv
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AutonomousVehicle.constants import *
from AutonomousVehicle.modeling.car import Car
from AutonomousVehicle.modeling.obstacles import Obstacles
from api.event_types import *
from .event_bus import EventBus

READ_FROM_FILE = True
METER_PER_PIXEL = 0.1
# MAP_FILE = _project_root / "AutonomousVehicle" / "map" / "map2.png"

def _generate_obstacles() -> npt.NDArray[np.floating[Any]]:
    """Generate obstacle coordinates programmatically"""
    ox = [
        np.arange(0, MAP_WIDTH, MAP_STEP),
        np.full(np.ceil(MAP_HEIGHT / MAP_STEP).astype(int), MAP_WIDTH),
        np.arange(0, MAP_WIDTH + MAP_STEP, MAP_STEP),
        np.full(np.ceil(MAP_HEIGHT / MAP_STEP).astype(int) + 1, 0.0),
        np.full(np.ceil(MAP_WIDTH / 3 * 2 / MAP_STEP).astype(int), MAP_WIDTH / 3),
        np.full(np.ceil(MAP_HEIGHT / 3 * 2 / MAP_STEP).astype(int), 2 * MAP_WIDTH / 3),
    ]
    oy = [
        np.full(np.ceil(MAP_WIDTH / MAP_STEP).astype(int), 0.0),
        np.arange(0, MAP_HEIGHT, MAP_STEP),
        np.full(np.ceil((MAP_WIDTH + MAP_STEP) / MAP_STEP).astype(int), MAP_HEIGHT),
        np.arange(0, MAP_HEIGHT + MAP_STEP, MAP_STEP),
        np.arange(0, MAP_WIDTH / 3 * 2, MAP_STEP),
        MAP_HEIGHT - np.arange(0, MAP_HEIGHT / 3 * 2, MAP_STEP),
    ]
    return np.vstack((np.concatenate(ox), np.concatenate(oy))).T


def _read_map(map_file: Path) -> npt.NDArray[np.floating[Any]]:
    """Read map obstacles from image file"""
    src = cv.imread(str(map_file), cv.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(f"Cannot read map file: {map_file}")
    src = cv.threshold(src, 127, 255, cv.THRESH_BINARY)[1]
    H, W = src.shape[:2]
    boundary = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    contours, _ = cv.findContours(src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    res = []
    for contour in chain(contours, [boundary]):
        if len(contour.shape) == 3:
            contour = contour[:, 0, :]
        if contour.shape[0] == 1:
            res.append(contour)
            continue
        contour = np.append(contour, contour[:1], axis=0)  # close the contour
        dists = np.linalg.norm(contour[:-1] - contour[1:], axis=1)
        u = np.concatenate(([0], np.cumsum(dists)))
        tck, _ = scipy.interpolate.splprep(contour.T, s=0, k=1, u=u)
        u = np.arange(0, u[-1], MAP_STEP / METER_PER_PIXEL)
        xy = np.column_stack(scipy.interpolate.splev(u, tck))
        res.append(xy)
    res = np.vstack(res)
    res[:, 1] = H - res[:, 1]  # flip y axis to match the image
    return res * METER_PER_PIXEL

class MapServerAdapter:
    """
    Adapter for MapServerNode
    
    Features:
    1. Load map obstacles (from file or generated programmatically)
    2. Manage dynamic obstacles (discovered via LIDAR scan)
    3. Publish obstacle update events via EventBus
    
    Events published (see api/event_types.py):
    - MAP_INITIALIZED: () - Map initialization complete
    - KNOWN_OBSTACLES_UPDATED: (coords: np.ndarray) - Known obstacles updated
    - NEW_OBSTACLES_DISCOVERED: (coords: np.ndarray) - New obstacles discovered
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._known_obstacle_coordinates = None
        self._unknown_obstacle_coordinates = None
        self._unknown_obstacles = None
        self._havent_discovered = None
        self._bounding_box = None
        self._map_file = None

    @property
    def known_obstacle_coordinates(self) -> npt.NDArray[np.floating[Any]] | None:
        return self._known_obstacle_coordinates
    
    @property
    def unknown_obstacle_coordinates(self) -> npt.NDArray[np.floating[Any]] | None:
        return self._unknown_obstacle_coordinates

    @property
    def bounding_box(self) -> tuple[float, float, float, float] | None:
        """Get map boundary (xmin, ymin, xmax, ymax)"""
        return self._bounding_box

    def init_map(self, map_name: str = "map2") -> None:
        """
        Initialize map data:

        1. Load map obstacles
        2. Generate random hidden obstacles
        3. Publish map initialization complete event

        Args:
            map_name: Map file name ("map" or "map2" or "map3" and etc), default "map2"
        """

        _project_root = Path(__file__).parent.parent.parent
        map_file = _project_root / "AutonomousVehicle" / "map" / f"{map_name}.png"
        self._map_file = map_file

        if not map_file.exists():
            raise FileNotFoundError(f"Map file not found: {map_file}")
        
        # 1. load or generate known obstacles
        self._known_obstacle_coordinates = coords = _read_map(map_file) if READ_FROM_FILE else _generate_obstacles()

        # 2. get map boundary
        xmin, ymin, xmax, ymax = coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()
        self._bounding_box = (xmin, ymin, xmax, ymax)

        # 3. generate randomized obstacles
        self._unknown_obstacle_coordinates = np.random.uniform(
            (xmin, ymin), (xmax, ymax), (MAP_NUM_RANDOM_OBSTACLES, 2)
        )
        self._unknown_obstacles = Obstacles(self._unknown_obstacle_coordinates)
        self._havent_discovered = np.ones(len(self._unknown_obstacle_coordinates), dtype=bool)

        # 4. publish event
        self._event_bus.emit(MAP_INITIALIZED)
        self._event_bus.emit(KNOWN_OBSTACLES_UPDATED, self._known_obstacle_coordinates)

    def _lidar_scan(self, x: float, y: float) -> None:
        """
        Simulate LIDAR scan to discover nearby hidden obstacles
        
        Args:
            x: x-coordinate of scan center
            y: y-coordinate of scan center
        """
        if (self._unknown_obstacles is None or 
            self._havent_discovered is None or 
            self._unknown_obstacle_coordinates is None or 
            self._known_obstacle_coordinates is None):
            return
        # 1. query obstacles within scan radius
        ids = np.array(self._unknown_obstacles.kd_tree.query_ball_point((x, y), Car.SCAN_RADIUS))
        if ids.size == 0:
            return 
        
        # 2. filter undiscovered obstacles
        ids: np.ndarray = ids[self._havent_discovered[ids]]
        if ids.size == 0:
            return
        
        # 3. mark as discovered
        self._havent_discovered[ids] = False
        new_obstacle_coordinates = self._unknown_obstacle_coordinates[ids]

        # 4. add to known obstacles
        self._known_obstacle_coordinates = np.vstack((self._known_obstacle_coordinates, new_obstacle_coordinates))

        # 5. publish events
        self._event_bus.emit(NEW_OBSTACLES_DISCOVERED, new_obstacle_coordinates)
        self._event_bus.emit(KNOWN_OBSTACLES_UPDATED, self._known_obstacle_coordinates)
    
    def update_from_vehicle_state(self, timestamp_s: float, state: Car) -> None:
        """
        Update map state based on vehicle position (called periodically from outside)
        
        This method is designed to be subscribed to MEASURED_STATE events.
        
        Args:
            timestamp_s: Simulation timestamp (seconds) - kept for event compatibility
            state: Current vehicle state
        """
        cy, sy = np.cos(state.yaw), np.sin(state.yaw)
        self._lidar_scan(state.x + cy * Car.BACK_TO_CENTER, state.y + sy * Car.BACK_TO_CENTER)
    
    def generate_random_initial_state(self) -> Car:
        """
        Generate a random collision-free initial vehicle state
        
        Returns:
            Car: Randomly generated vehicle state
        """
        if (self._known_obstacle_coordinates is None or 
            self._unknown_obstacle_coordinates is None):
            raise RuntimeError("Map must be initialized before generating random initial state. Call init_map() first.")
        
        obstacles = Obstacles(np.vstack((self._known_obstacle_coordinates, self._unknown_obstacle_coordinates)))
        state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
        while Car(*state).check_collision(obstacles):
            state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
        return Car(*state)