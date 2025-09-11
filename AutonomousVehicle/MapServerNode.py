from itertools import chain
from pathlib import Path
from typing import Any, Optional

import cv2 as cv
import numpy as np
import numpy.typing as npt
import scipy.interpolate
from PySide6.QtCore import QObject, Signal, Slot

from .constants import *
from .modeling.car import Car
from .modeling.obstacles import Obstacles

READ_FROM_FILE = True
MAP_FILE = Path(__file__).absolute().parent / "map" / "map2.png"
METER_PER_PIXEL = 0.1


def _generate_obstacles() -> npt.NDArray[np.floating[Any]]:
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


def _read_map() -> npt.NDArray[np.floating[Any]]:
    src = cv.imread(str(MAP_FILE), cv.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(f"Cannot read map file: {MAP_FILE}")
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


class MapServerNode(QObject):
    known_obstacle_coordinates_updated = Signal(np.ndarray)
    new_obstacle_coordinates = Signal(np.ndarray)
    inited = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

    @Slot()
    def init(self) -> None:
        self._known_obstacle_coordinates = coords = _read_map() if READ_FROM_FILE else _generate_obstacles()
        xmin, ymin, xmax, ymax = coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()
        self._bounding_box = (xmin, ymin, xmax, ymax)
        self._unknown_obstacle_coordinates = np.random.uniform(
            (xmin, ymin), (xmax, ymax), (MAP_NUM_RANDOM_OBSTACLES, 2)
        )
        self._unknown_obstacles = Obstacles(self._unknown_obstacle_coordinates)
        self._havent_discovered = np.ones(len(self._unknown_obstacle_coordinates), dtype=bool)
        self.inited.emit()
        self.known_obstacle_coordinates_updated.emit(self._known_obstacle_coordinates)

    @property
    def known_obstacle_coordinates(self) -> npt.NDArray[np.floating[Any]]:
        return self._known_obstacle_coordinates

    @property
    def unknown_obstacle_coordinates(self) -> npt.NDArray[np.floating[Any]]:
        return self._unknown_obstacle_coordinates

    def _lidar_scan(self, x: float, y: float) -> None:
        ids = np.array(self._unknown_obstacles.kd_tree.query_ball_point((x, y), Car.SCAN_RADIUS))
        if ids.size == 0:
            return
        ids: np.ndarray = ids[self._havent_discovered[ids]]
        if ids.size == 0:
            return
        self._havent_discovered[ids] = False
        new_obstacle_coordinates = self._unknown_obstacle_coordinates[ids]
        self._known_obstacle_coordinates = np.vstack((self._known_obstacle_coordinates, new_obstacle_coordinates))
        self.new_obstacle_coordinates.emit(new_obstacle_coordinates)
        self.known_obstacle_coordinates_updated.emit(self._known_obstacle_coordinates)

    @Slot(float, Car)
    def update(self, timestamp_s: float, state: Car) -> None:
        cy, sy = np.cos(state.yaw), np.sin(state.yaw)
        self._lidar_scan(state.x + cy * Car.BACK_TO_CENTER, state.y + sy * Car.BACK_TO_CENTER)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return self._bounding_box

    def generate_random_initial_state(self) -> Car:
        obstacles = Obstacles(np.vstack((self._known_obstacle_coordinates, self._unknown_obstacle_coordinates)))
        state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
        while Car(*state).check_collision(obstacles):
            state = np.random.uniform((0, 0, -np.pi), (MAP_WIDTH, MAP_HEIGHT, np.pi))
        return Car(*state)
