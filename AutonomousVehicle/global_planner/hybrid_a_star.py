import heapq
from itertools import islice, product

from typing import Any, Generator, Literal, NamedTuple, Optional


import numpy as np
import numpy.typing as npt
from rsplan import Path as RSPath

from ..constants import *
from ..modeling.Car import Car
from ..modeling.Obstacles import ObstacleGrid, Obstacles

from ..utils.wrap_angle import wrap_angle

XY_GRID_RESOLUTION = 1.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_DISTANCE = XY_GRID_RESOLUTION * 1.5  # [m] path interpolate distance
NUM_STEER_COMMANDS = 10  # number of steer command

SWITCH_DIRECTION_COST = 25.0  # switch direction cost
BACKWARDS_COST = 4.0  # backward movement cost
STEER_CHANGE_COST = 3.0  # steer angle change cost
STEER_COST = 1.5  # steer angle cost per distance
H_DIST_COST = 2.0  # Heuristic distance cost
H_YAW_COST = 3.0 / np.deg2rad(45)  # Heuristic yaw difference cost
H_COLLISION_COST = 1e4  # collision cost when calculating heuristic


STEER_COMMANDS = np.unique(
    np.concatenate([np.linspace(-Car.TARGET_MAX_STEER, Car.TARGET_MAX_STEER, NUM_STEER_COMMANDS), [0.0]])
)


MOVEMENTS = tuple(
    # di, dj, cost
    (di, dj, np.sqrt(di**2 + dj**2)) for di in (-1, 0, 1) for dj in(-1, 0, 1) if di or dj
)

def _distance_heuristic(grid: ObstacleGrid, goal_xy: npt.ArrayLike) -> ObstacleGrid:
    "Dijkstra's algorithm to calculate the distance from each grid cell to the goal"
    H, W = grid.grid.shape
    dist = np.full((H, W), H_COLLISION_COST)
    ij = grid.calc_index(goal_xy)
    dist[ij] = 0
    pq = [(0, ij)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, cost in MOVEMENTS:
            ni, nj, n_cost = i + di, j + dj, d + cost
            if 0 <= ni < H and 0 <= nj < W and not grid.grid[ni, nj] and n_cost < dist[ni, nj]:
                dist[ni, nj] = n_cost
                heapq.heappush(pq, (n_cost, (ni, nj)))
    return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)



class SimplePath(NamedTuple):
    ijk: tuple[int, int, int]  # grid index
    trajectory: npt.NDArray[np.floating[Any]]  # [[x(m), y(m), yaw(rad)]]
    direction: Literal[1, 0, -1]  # direction, 1 forward, -1 backward, 0 initial
    steer: float  # [rad], [-TARGET_MAX_STEER, TARGET_MAX_STEER]

class Node(NamedTuple):
    path: SimplePath | RSPath
    cost: float  # cost from start to this node
    h_cost: float  # h_cost from node to the goal
    parent: Optional["Node"]



def hybrid_a_star(
        start: npt.NDArray[np.floating[Any]],
        goal: npt.NDArray[np.floating[Any]],
        obstacles: Obstacles
    ) -> Optional[npt.NDArray[np.floating[Any]]]:

    # ================= Precheck Start/Goal Collision =================
    if Car(*goal).check_collision(obstacles):
        return None
    start_is_point = start.shape == (3,)
    start_collided = Car(*start).check_collision(obstacles) if start_is_point else False
    
    # ================= Heuristic Grid =================
    # 1. Downsample the obstacles to a grid
    obstacle_grid = obstacles.downsampling_to_grid(
        XY_GRID_RESOLUTION, min(Car.COLLISION_LENGTH, Car.COLLISION_WIDTH) / 2
    )
    # 2. Precompute the distance to the goal from each grid cell, where the distance will be used as a heuristic
    heuristic_grid = _distance_heuristic(obstacle_grid, goal[:2])
    N, M = heuristic_grid.grid.shape
    K = int(2 * np.pi / YAW_GRID_RESOLUTION)

    # Used to record the path and cost for each grid cell at A* search stage,
    # where dp[y][x][yaw] is the Node object for the grid cell (x, y) with yaw angle yaw
    dp = np.full((N, M, K), None, dtype=Node)
    
    def calc_ijk(x: float, y: float, yaw: float) -> tuple[int, int, int]:
        "Map ObstacleGrid [x, y, yaw] -> Heuristic [i, j, k] for dp"
        i, j = heuristic_grid.calc_index([x, y])
        k = int(wrap_angle(yaw, zero_to_2pi=True) // YAW_GRID_RESOLUTION)
        return i, j, k

    def generate_neighbor(curr: Node, direction: int, steer: float) -> Optional[Node]:
        "Generate a neighbour node of the current node, given the direction and steer angle"
        
        # Simulate the car movement for MOTION_DISTANCE, with a interval of MOTION_RESOLUTION,
        # check if the car will collide with the obstacles during the movement
        car = Car(*curr.path.trajectory[-1, :3], velocity=float(direction), steer=steer)
        trajectory = []
        for _ in range(int(MOTION_DISTANCE / MOTION_RESOLUTION)):
            car.update(MOTION_RESOLUTION)
            if not start_collided and car.check_collision(obstacles):
                return None
            trajectory.append([car.x, car.y, car.yaw])

        i, j, k = calc_ijk(car.x, car.y, car.yaw)
        if not (0 <= i < N and 0 <= j < M):
            print(f"Out of grid, please add more obstacles to fill the boundary: {i=} {j=}")
            return None
        
        # Calculate the cost from the start to this neighbour node
        distance_cost = MOTION_DISTANCE if direction == 1 else MOTION_DISTANCE * BACKWARDS_COST
        switch_direction_cost = (
            SWITCH_DIRECTION_COST if curr.path.direction != 0 and direction != curr.path.direction 
            else 0.0
        )
        steer_change_cost = STEER_CHANGE_COST * abs(steer - curr.path.steer)
        steer_cost = STEER_COST * abs(steer) * MOTION_DISTANCE
        
        cost = curr.cost + distance_cost + switch_direction_cost + steer_change_cost + steer_cost

        # Calculate the heuristic cost fromm this neighbor node to the goal
        h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]
        h_yaw_cost = H_YAW_COST * abs(wrap_angle(goal[2] - car.yaw))
        h_cost = h_dist_cost + h_yaw_cost

        return Node(SimplePath((i, j, k), np.array(trajectory), direction, steer), cost, h_cost, curr)

    def generate_neighbors(curr: Node) -> Generator[Node, None, Node]:
        "Generate all possible neighbours of the current node"
        nonlocal start_collided
        for direction, steer in product([1, -1], STEER_COMMANDS):
            if (node := generate_neighbor(curr, direction, steer)) is not None:
                yield node
        start_collided = False