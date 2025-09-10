import math, heapq
from collections.abc import Callable, Generator
from typing import Any, Literal, NamedTuple, Optional
from itertools import islice, product

import numpy as np
import numpy.typing as npt
from rsplan import Path as RSPath

from ..constants import *
from ..utils.wrap_angle import wrap_angle
from ..utils.SupportsBool import SupportsBool
from modeling.car import Car
from modeling.obstacles import ObstacleGrid, Obstacles

XY_GRID_RESOLUTION = 1.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_DISTANCE = XY_GRID_RESOLUTION * 1.5  # [m] path interpolate distance
NUM_STEER_COMMANDS = 10  # number of steer command

REEDS_SHEPP_MAX_DISTANCE = 10.0 

BACKWARDS_COST = 4.0  # backward movement cost
SWITCH_DIRECTION_COST = 25.0  # switch direction cost
STEER_CHANGE_COST = 3.0  # steer angle change cost
STEER_COST = 1.5  # steer angle cost per distance
H_DIST_COST = 2.0  # Heuristic distance cost
H_YAW_COST = 3.0 / np.deg2rad(45)  # Heuristic yaw difference cost
H_COLLISION_COST = 1e4  # collision cost when calculating heuristic

# if True, return the Reeds-Shepp path immediately when it is found
# otherwise, continue the A* search to find a better path (may be much slower)
RETURN_RS_PATH_IMMEDIATELY = True


STEER_COMMANDS = np.unique(
    np.concatenate([np.linspace(-Car.TARGET_MAX_STEER, Car.TARGET_MAX_STEER, NUM_STEER_COMMANDS), [0.0]])
)

MOVEMENTS = tuple((di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)


class SimplePath(NamedTuple):
    ijk: tuple[int, int, int]  # grid index
    trajectory: npt.NDArray[np.floating[Any]]  # [[x(m), y(m), yaw(rad)]]
    direction: Literal[1, 0, -1]  # direction, 1 forward, -1 backward, 0 initial
    steer: float  # [rad], [-TARGET_MAX_STEER, TARGET_MAX_STEER]

class Node(NamedTuple):
    path: SimplePath | RSPath
    cost: float
    h_cost: float
    parent: Optional["Node"]

    def __lt__(self, other: "Node") -> bool:
        # f = h(dijkstra) + g(actual) 
        return (self.h_cost + self.cost, self.cost) < (other.h_cost + other.cost, other.cost)

    def get_plot_trajectory(self) -> npt.NDArray[np.floating[Any]]:
        "Get the trajectory coordinates for visualization"
        trajectory = (
            np.array([[p.x, p.y] for p in self.path.waypoints()])
            if isinstance(self.path, RSPath)
            else self.path.trajectory[:, :2]
        )
        if self.parent is not None:
            trajectory = np.vstack((self.parent.path.trajectory[-1, :2], trajectory))
        return trajectory
    

## Distance Field Calculation based on Dijkstra Algorithm
def _heuristic_distance_field(grid: ObstacleGrid, goal_xy: tuple[float, float]) -> ObstacleGrid:
    H, W = grid.grid.shape
    dist = np.full((H, W), H_COLLISION_COST)

    gi, gj = grid.calc_index(goal_xy)
    if not (0 <= gi < H and 0 <= gj < W):
        return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)
    if grid.grid[gi, gj]:
        return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)
    
    dist[gi, gj] = 0.0
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, (gi, gj))]

    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, step in MOVEMENTS:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not grid.grid[ni, nj] and d + step < dist[ni, nj]:
                dist[ni, nj] = d + step
                heapq.heappush(pq, (d + step, (ni, nj)))

    return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)

def hybrid_a_star(start: npt.NDArray[np.floating[Any]],
                  goal: npt.NDArray[np.floating[Any]],
                  obstacles: Obstacles,
                  cancel_callback: Optional[Callable[[Node], SupportsBool]] = None
                ) -> Optional[npt.NDArray[np.floating[Any]]]:
    
    assert start.shape == (3,) or (
        len(start.shape) == 2 and start.shape[1] == 4
        ), "Start must be a 1D array of shape (3) representing [x, y, yaw] or a 2D array of shape (N, 4) representing [x, y, yaw, velocity]"
    
    assert goal.shape == (3,), "Goal must be a 1D array of shape (3) representing [x, y, yaw]"

    if Car(*goal).check_collision(obstacles):
        return None
    
    start_is_point = start.shape == (3,)
    start_collided = Car(*start).check_collision(obstacles) if start_is_point else False

    # Downsample the obstacle map to a grid
    obstacle_grid = obstacles.downsampling_to_grid(
        XY_GRID_RESOLUTION, 
        min(Car.COLLISION_LENGTH, Car.COLLISION_WIDTH)/2 # r/2
        )
    
    # Compute Dijkstra Heuristic Field, each grid cell obtain obtimal distance to goal. Will be used for f = g + h search
    heuristic_grid = _heuristic_distance_field(obstacle_grid, goal[:2])
    N, M = heuristic_grid.grid.shape
    # K = int(2 * np.pi * YAW_GRID_RESOLUTION)
    K = int(round(2 * np.pi / YAW_GRID_RESOLUTION))
    

    ### ================== helpers ================== ### 

    def _calc_ijk(x: float, y: float, yaw: float) -> tuple[int, int, int]:
        "[x, y, yaw] -> [i, j, k] for dp"
        i, j = heuristic_grid.calc_index([x, y])
        k = int(wrap_angle(yaw, zero_to_2pi=True) // YAW_GRID_RESOLUTION)
        return i, j, k
    
    def rollout(curr: Node, direction: int, steer: float) -> Optional[Node]:
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

        i, j, k = _calc_ijk(car.x, car.y, car.yaw)
        if not (0 <= i < N and 0 <= j < M):
            print(f"Out of grid, please add more obstacles to fill the boundary: {i=} {j=}")
            return None

        # Calculate the cost from the start to this neighbour node
        distance_cost = MOTION_DISTANCE if direction == 1 else MOTION_DISTANCE * BACKWARDS_COST
        switch_direction_cost = (
            SWITCH_DIRECTION_COST if curr.path.direction != 0 and direction != curr.path.direction else 0.0
        )
        steer_change_cost = STEER_CHANGE_COST * abs(steer - curr.path.steer)
        steer_cost = STEER_COST * abs(steer) * MOTION_DISTANCE
        cost = curr.cost + distance_cost + switch_direction_cost + steer_change_cost + steer_cost

        # Calculate the heuristic cost from this neighbour node to the goal
        h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]
        h_yaw_cost = H_YAW_COST * abs(wrap_angle(goal[2] - car.yaw))
        h_cost = h_dist_cost + h_yaw_cost

        return Node(SimplePath((i, j, k), np.array(trajectory), direction, steer), cost, h_cost, curr)
    
    def _generate_neighbors(curr: Node,) -> Generator[Node, None, None]:
        "Generate all possible neighbours of the current node"
        nonlocal start_collided
        for direction, steer in product([1, -1], STEER_COMMANDS):
            if (res := rollout(curr, direction, steer)) is not None:
                yield res
        start_collided = False
        return 
    
    def _reached_goal(x: float, y:float, yaw:float) -> bool:
        pos_ok = np.hypot(x - goal[0], y - goal[1]) <= XY_GRID_RESOLUTION * 1.0
        yaw_ok = abs(wrap_angle(yaw - goal[2])) <= np.deg2rad(45)
        return pos_ok and yaw_ok
    
    def _generate_rspath(node: Node) -> Optional[Node]:
        """
        Try to generate a Path from the current node directly to the goal using Reeds-Shepp curves,
        which will speed up the search process when the node is close to the goal and heuristics
        are not enough to guide the search.
        """
        return
    
    def _reconstruct_path(node: Node) -> npt.NDArray[np.floating[Any]]:
        """
        Traceback the path from the goal to the start, to get the final trajectory

        returns [[x(m), y(m), yaw(rad), direction(1, -1)]]
        """
        return
    
    def _end_pose(n:Node) -> tuple[float, float, float]:
        if isinstance(n.path, RSPath):
            last_waypoint = n.path.waypoints()[-1]
            return last_waypoint.x, last_waypoint.y, goal[2]
        else:
            x, y, yaw = n.path.trajectory[-1, :3]
            return float(x), float(y), float(yaw)
        
    ### ================== MAIN LOOP ================== ### 

    if start.shape == (3,):
        start_i, start_j, start_k = _calc_ijk(start[0], start[1], start[2])
        start_traj = np.array([[start[0], start[1], start[2]]], float)
        start_dir, start_steer = 0, 0.0
    else:
        # remove replicated
        xy = start[:, :2]
        mask = (xy[:-1] != xy[1:]).any(axis=1)
        start = start[np.concatenate(([True], mask))]

        start_i, start_j, start_k = _calc_ijk(*start[-1, :3])  # last gesture ijk

        # Normalize the "direction column" to retain only the sign (+1/-1)
        start[:, 3] = np.sign(start[0, 3])
        start_dir = int(start[0, 3])

        # Estimate the initial steering angle (based on the change in orientation/arc length of the last two frames)
        if start.shape[0] >= 2:
            l = np.linalg.norm(start[-1, :2] - start[-2, :2])
            start_steer = np.arctan(Car.WHEEL_BASE * (start[-1, 2] - start[-2, 2]) / l) if l > 1e-6 else 0.0
        else:
            start_steer = 0.0
        start_traj = start[:, :3]
        

    start_path = SimplePath(
        (start_i, start_j, start_k), 
        start_traj, 
        start_dir, 
        start_steer
    )
    
    start_h = H_DIST_COST * heuristic_grid.grid[start_i, start_j] + H_YAW_COST * abs(wrap_angle(goal[2] - start_traj[-1,2]))

    start_node = Node(start_path, 0.0, start_h, None)

    ### Initialize A* search state ###
    # Open set (min-heap priority queue, Node.__lt__ sorts by f=g+h for A* search)
    pq: list[Node] = [start_node]
    heapq.heapify(pq)

    dp = np.empty((N, M, K), dtype=object)  # 与原作一致：dp[(i,j,k)] 存“当前最优的 Node”
    dp[:] = None
    dp[start_i, start_j, start_k] = start_node

    while pq:
        curr = heapq.heappop(pq)

        if cancel_callback is not None and bool(cancel_callback(curr)):
            return None
        

        curr_best = dp[curr.path.ijk]
        if (curr_best is not None) and (curr.cost > curr_best.cost):
            continue

        # Termination condition
        # Near-end option: 
        # resolve direct connection (Reeds–Shepp), use it as the final segment if successful
        cx, cy, cyaw = _end_pose(curr)
        if _reached_goal(cx, cy, cyaw):
            rs_last = _generate_rspath(curr)
            if rs_last is not None:
                return _reconstruct_path(rs_last)
            return _reconstruct_path(curr)
        
        # When close to the goal, first try Reeds–Shepp direct connection and push into the heap to speed up
        if np.linalg.norm(curr.path.trajectory[-1, :2] - goal[:2]) <= REEDS_SHEPP_MAX_DISTANCE:
            if (rsnode := _generate_rspath(curr)) is not None:
                if RETURN_RS_PATH_IMMEDIATELY:
                    return _reconstruct_path(rsnode)
                heapq.heappush(pq, rsnode)

        for child in _generate_neighbors(curr):
            if dp[child.path.ijk] is None or child.cost < dp[child.path.ijk].cost:
                dp[child.path.ijk] = child
                heapq.heappush(pq, child)
        
    return None