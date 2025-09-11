# import math, heapq
# from collections.abc import Callable, Generator
# from typing import Any, Literal, NamedTuple, Optional
# # from itertools import islice, product

# import numpy as np
# import numpy.typing as npt
# from rsplan import Path as RSPath
# from rsplan.planner import _solve_path as solve_rspath

# from ..constants import *
# from ..utils.wrap_angle import wrap_angle
# from ..utils.SupportsBool import SupportsBool
# from ..modeling.car import Car
# from ..modeling.obstacles import ObstacleGrid, Obstacles

# XY_GRID_RESOLUTION = 1.0  # [m]
# YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
# MOTION_DISTANCE = XY_GRID_RESOLUTION * 1.5  # [m] path interpolate distance
# NUM_STEER_COMMANDS = 10  # number of steer command

# REEDS_SHEPP_MAX_DISTANCE = 10.0 

# BACKWARDS_COST = 4.0  # backward movement cost
# SWITCH_DIRECTION_COST = 25.0  # switch direction cost
# STEER_CHANGE_COST = 3.0  # steer angle change cost
# STEER_COST = 1.5  # steer angle cost per distance
# H_DIST_COST = 2.0  # Heuristic distance cost
# H_YAW_COST = 3.0 / np.deg2rad(45)  # Heuristic yaw difference cost
# H_COLLISION_COST = 1e4  # collision cost when calculating heuristic

# # if True, return the Reeds-Shepp path immediately when it is found
# # otherwise, continue the A* search to find a better path (may be much slower)
# RETURN_RS_PATH_IMMEDIATELY = True


# STEER_COMMANDS = np.unique(
#     np.concatenate([np.linspace(-Car.TARGET_MAX_STEER, Car.TARGET_MAX_STEER, NUM_STEER_COMMANDS), [0.0]])
# )

# MOVEMENTS = tuple((di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)


# class SimplePath(NamedTuple):
#     ijk: tuple[int, int, int]  # grid index
#     trajectory: npt.NDArray[np.floating[Any]]  # [[x(m), y(m), yaw(rad)]]
#     direction: Literal[1, 0, -1]  # direction, 1 forward, -1 backward, 0 initial
#     steer: float  # [rad], [-TARGET_MAX_STEER, TARGET_MAX_STEER]

# class Node(NamedTuple):
#     path: SimplePath | RSPath
#     cost: float
#     h_cost: float
#     parent: Optional["Node"]

#     def __lt__(self, other: "Node") -> bool:
#         # f = h(dijkstra) + g(actual) 
#         return (self.h_cost + self.cost, self.cost) < (other.h_cost + other.cost, other.cost)

#     def get_plot_trajectory(self) -> npt.NDArray[np.floating[Any]]:
#         "Get the trajectory coordinates for visualization"
#         trajectory = (
#             np.array([[p.x, p.y] for p in self.path.waypoints()])
#             if isinstance(self.path, RSPath)
#             else self.path.trajectory[:, :2]
#         )
#         if self.parent is not None:
#             trajectory = np.vstack((self.parent.path.trajectory[-1, :2], trajectory))
#         return trajectory
    

# ## Distance Field Calculation based on Dijkstra Algorithm
# def _heuristic_distance_field(grid: ObstacleGrid, goal_xy: tuple[float, float]) -> ObstacleGrid:
#     H, W = grid.grid.shape
#     dist = np.full((H, W), H_COLLISION_COST)

#     gi, gj = grid.calc_index(goal_xy)
#     if not (0 <= gi < H and 0 <= gj < W):
#         return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)
#     if grid.grid[gi, gj]:
#         return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)
    
#     dist[gi, gj] = 0.0
#     pq: list[tuple[float, tuple[int, int]]] = [(0.0, (gi, gj))]

#     while pq:
#         d, (i, j) = heapq.heappop(pq)
#         if d > dist[i, j]:
#             continue
#         for di, dj, step in MOVEMENTS:
#             ni, nj = i + di, j + dj
#             if 0 <= ni < H and 0 <= nj < W and not grid.grid[ni, nj] and d + step < dist[ni, nj]:
#                 dist[ni, nj] = d + step
#                 heapq.heappush(pq, (d + step, (ni, nj)))

#     return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)

# def hybrid_a_star(start: npt.NDArray[np.floating[Any]],
#                   goal: npt.NDArray[np.floating[Any]],
#                   obstacles: Obstacles,
#                   cancel_callback: Optional[Callable[[Node], SupportsBool]] = None
#                 ) -> Optional[npt.NDArray[np.floating[Any]]]:
    
#     assert start.shape == (3,) or (
#         len(start.shape) == 2 and start.shape[1] == 4
#         ), "Start must be a 1D array of shape (3) representing [x, y, yaw] or a 2D array of shape (N, 4) representing [x, y, yaw, velocity]"
    
#     assert goal.shape == (3,), "Goal must be a 1D array of shape (3) representing [x, y, yaw]"

#     if Car(*goal).check_collision(obstacles):
#         return None
    
#     # start_is_point = start.shape == (3,)
#     # start_collided = Car(*start).check_collision(obstacles) if start_is_point else False

#     # Downsample the obstacle map to a grid
#     obstacle_grid = obstacles.downsampling_to_grid(
#         XY_GRID_RESOLUTION, 
#         min(Car.COLLISION_LENGTH, Car.COLLISION_WIDTH)/2 # r/2
#         )
    
#     # Compute Dijkstra Heuristic Field, each grid cell obtain obtimal distance to goal. Will be used for f = g + h search
#     heuristic_grid = _heuristic_distance_field(obstacle_grid, goal[:2])
#     N, M = heuristic_grid.grid.shape
#     # K = int(2 * np.pi * YAW_GRID_RESOLUTION)
#     K = int(round(2 * np.pi / YAW_GRID_RESOLUTION))
    

#     ### ================== helpers ================== ### 

#     def _calc_ijk(x: float, y: float, yaw: float) -> tuple[int, int, int]:
#         "[x, y, yaw] -> [i, j, k] for dp"
#         i, j = heuristic_grid.calc_index([x, y])
#         k = int(wrap_angle(yaw, zero_to_2pi=True) // YAW_GRID_RESOLUTION)
#         return i, j, k
    
#     def rollout(curr: Node, direction: int, steer: float) -> Optional[Node]:
#         "Generate a neighbour node of the current node, given the direction and steer angle"

#         # Simulate the car movement for MOTION_DISTANCE, with a interval of MOTION_RESOLUTION,
#         # check if the car will collide with the obstacles during the movement

#         x, y, yaw = curr.path.trajectory[-1, :3]

#         car = Car(float(x), float(y), float(yaw), velocity=1.0*direction, steer=float(steer))
#         steps = int(math.ceil(MOTION_DISTANCE / MOTION_RESOLUTION))

#         trajectory = []
      
#         for _ in range(steps):
#             car.update(MOTION_RESOLUTION, do_wrap_angle=True)
#             if car.check_collision(obstacles):
#                 return None
#             trajectory.append([car.x, car.y, car.yaw])
#         trajectory = np.asarray(trajectory, float)

#         # 离散 + 出界/占据
#         i, j, k = _calc_ijk(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2])
#         if not (0 <= i < N and 0 <= j < M):
#             # print(f"Out of grid: i={i}, j={j}")
#             return None
#         if obstacle_grid.grid[i, j]:
#             return None

#         # Calculate the cost from the start to this neighbour node
#         distance_cost = MOTION_DISTANCE if direction == 1 else MOTION_DISTANCE * BACKWARDS_COST
#         switch_direction_cost = (
#             SWITCH_DIRECTION_COST if curr.path.direction != 0 and direction != curr.path.direction else 0.0
#         )
#         # steer_change_cost = STEER_CHANGE_COST * abs(steer - curr.path.steer)
#         steer_change_cost = STEER_CHANGE_COST * (
#             abs(steer - curr.path.steer) if curr.path.direction != 0 else 0.0
#         )
#         steer_cost = STEER_COST * abs(steer) * MOTION_DISTANCE
#         cost = curr.cost + distance_cost + switch_direction_cost + steer_change_cost + steer_cost

#         # Calculate the heuristic cost from this neighbour node to the goal
#         h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]
#         h_yaw_cost = H_YAW_COST * abs(wrap_angle(goal[2] - car.yaw))
#         h_cost = h_dist_cost + h_yaw_cost

#         return Node(SimplePath((i, j, k), np.array(trajectory), direction, steer), cost, h_cost, curr)
    
#     def _generate_neighbors(curr: Node,) -> Generator[Node, None, None]:
#         "Generate all possible neighbours of the current node"
#         for direction in (+1, -1):
#             for steer in STEER_COMMANDS:
#                 child = rollout(curr, int(direction), float(steer))
#                 if child is not None:
#                     yield child
    
#     def _reached_goal(x: float, y:float, yaw:float) -> bool:
#         pos_ok = np.hypot(x - goal[0], y - goal[1]) <= XY_GRID_RESOLUTION * 1.0
#         yaw_ok = abs(wrap_angle(yaw - goal[2])) <= np.deg2rad(45)
#         return pos_ok and yaw_ok
    
#     def _generate_rspath(node: Node) -> Optional[Node]:
#         def check(path: RSPath) -> bool:
#             for x, y, yaw in zip(*path.coordinates_tuple()):
#                 if Car(x, y, yaw).check_collision(obstacles):
#                     return False
#             return True

#         def calc_rspath_cost(path: RSPath) -> float:
#             last_direction = node.path.direction if not isinstance(node.path, RSPath) else 0
#             last_steer = node.path.steer if not isinstance(node.path, RSPath) else 0.0

#             distance_cost = 0.0
#             switch_direction_cost = 0.0
#             steer_change_cost = 0.0
#             steer_cost = 0.0

#             for seg in path.segments:
#                 length = abs(seg.length)
#                 distance_cost += length if seg.direction == 1 else length * BACKWARDS_COST
#                 if last_direction != 0 and seg.direction != last_direction:
#                     switch_direction_cost += SWITCH_DIRECTION_COST
#                 last_direction = seg.direction

#                 steer = {"left": Car.TARGET_MAX_STEER,
#                         "right": -Car.TARGET_MAX_STEER,
#                         "straight": 0.0}.get(seg.type, 0.0)
#                 steer_change_cost += STEER_CHANGE_COST * abs(steer - last_steer)
#                 last_steer = steer
#                 steer_cost += STEER_COST * abs(steer) * length

#             return distance_cost + switch_direction_cost + steer_change_cost + steer_cost

#         # 起点姿态用 _end_pose，兼容 RSPath/SimplePath
#         sx, sy, syaw = _end_pose(node)

#         # 生成所有 Reeds–Shepp 候选
#         paths = solve_rspath(
#             (sx, sy, syaw), tuple(goal),
#             Car.TARGET_MIN_TURNING_RADIUS,
#             MOTION_RESOLUTION
#         )

#         # 过滤碰撞
#         paths = filter(check, paths)
#         # 计算代价并取最优
#         best = None
#         best_cost = None
#         for p in paths:
#             c = calc_rspath_cost(p)
#             if (best is None) or (c < best_cost):
#                 best, best_cost = p, c
#         if best is None:
#             return None

#         return Node(best, node.cost + best_cost, 0.0, node)

#     def _ensure_dir_sign(traj: np.ndarray) -> np.ndarray:
#         """
#         输入 [N,3] 或 [N,4]（x,y,yaw[,dir]），输出 [N,4]，把 dir 统一成 ±1，绝不为 0。
#         规则：用相邻位移在朝向上的投影判断是前进(+1)还是倒车(-1)；无法判断时沿用前一个；第一个默认 +1。
#         """
#         assert traj.ndim == 2 and traj.shape[1] in (3, 4)
#         if traj.shape[1] == 3:
#             traj = np.hstack([traj, np.zeros((traj.shape[0], 1), float)])

#         dirc = traj[:, 3].astype(float).copy()
#         N = traj.shape[0]
#         for k in range(1, N):
#             if dirc[k] == 0:
#                 dx = traj[k, 0] - traj[k-1, 0]
#                 dy = traj[k, 1] - traj[k-1, 1]
#                 yaw = traj[k, 2]
#                 proj = dx * math.cos(yaw) + dy * math.sin(yaw)
#                 if proj > 1e-6:
#                     dirc[k] = 1.0
#                 elif proj < -1e-6:
#                     dirc[k] = -1.0
#                 else:
#                     dirc[k] = dirc[k-1] if dirc[k-1] != 0 else 1.0
#         if dirc[0] == 0:
#             # 如果第一个还没定，就看第一个非零，否则默认前进
#             nz = np.flatnonzero(dirc)
#             dirc[0] = dirc[nz[0]] if nz.size else 1.0

#         traj[:, 3] = np.sign(dirc)
#         return traj


#     def _reconstruct_path(node: Node) -> npt.NDArray[np.floating[Any]]:
#         segs = []
#         cur = node
#         while cur is not None:
#             if isinstance(cur.path, RSPath):
#                 xs, ys, yaws = cur.path.coordinates_tuple()  
#                 pts = np.stack([xs, ys, yaws], axis=1)
                
#                 d = cur.parent.path.direction if cur.parent and not isinstance(cur.parent.path, RSPath) else 1
#                 dircol = np.full((pts.shape[0], 1), d, float)
#                 segs.append(np.hstack([pts, dircol]))
#             else:
#                 traj = cur.path.trajectory  # [N,3]
#                 dircol = np.full((traj.shape[0], 1), cur.path.direction, float)
#                 segs.append(np.hstack([traj, dircol]))
#             cur = cur.parent
#         segs.reverse()
#         return _ensure_dir_sign(np.vstack(segs))

#     def _end_pose(n:Node) -> tuple[float, float, float]:
#         if isinstance(n.path, RSPath):
#             # last_waypoint = n.path.waypoints()[-1]
#             # return last_waypoint.x, last_waypoint.y, goal[2]
#             xs, ys, yaws = n.path.coordinates_tuple()
#             return float(xs[-1]), float(ys[-1]), float(yaws[-1])
#         else:
#             x, y, yaw = n.path.trajectory[-1, :3]
#             return float(x), float(y), float(yaw)
        
#     ### ================== MAIN LOOP ================== ### 

#     if start.shape == (3,):
#         start_i, start_j, start_k = _calc_ijk(start[0], start[1], start[2])
#         start_traj = np.array([[start[0], start[1], start[2]]], float)
#         start_dir, start_steer = 0, 0.0
#     else:
#         # remove replicated
#         xy = start[:, :2]
#         mask = (xy[:-1] != xy[1:]).any(axis=1)
#         start = start[np.concatenate(([True], mask))]

#         start_i, start_j, start_k = _calc_ijk(*start[-1, :3])  # last gesture ijk

#         # Normalize the "direction column" to retain only the sign (+1/-1)
#         start[:, 3] = np.sign(start[0, 3])
#         start_dir = int(start[0, 3])

#         # Estimate the initial steering angle (based on the change in orientation/arc length of the last two frames)
#         if start.shape[0] >= 2:
#             l = np.linalg.norm(start[-1, :2] - start[-2, :2])
#             start_steer = np.arctan(Car.WHEEL_BASE * (start[-1, 2] - start[-2, 2]) / l) if l > 1e-6 else 0.0
#         else:
#             start_steer = 0.0
#         start_traj = start[:, :3]
        

#     start_path = SimplePath(
#         (start_i, start_j, start_k), 
#         start_traj, 
#         start_dir, 
#         start_steer
#     )
    
#     start_h = H_DIST_COST * heuristic_grid.grid[start_i, start_j] + H_YAW_COST * abs(wrap_angle(goal[2] - start_traj[-1,2]))

#     start_node = Node(start_path, 0.0, start_h, None)

#     ### Initialize A* search state ###
#     # Open set (min-heap priority queue, Node.__lt__ sorts by f=g+h for A* search)
#     pq: list[Node] = [start_node]
#     heapq.heapify(pq)

#     dp = np.empty((N, M, K), dtype=object)  
#     dp[:] = None
#     dp[start_i, start_j, start_k] = start_node

#     while pq:
#         curr = heapq.heappop(pq)
        
#         if isinstance(curr.path, RSPath):
#             if cancel_callback is not None and bool(cancel_callback(curr)):
#                 return None
#             return _reconstruct_path(curr)

#         if cancel_callback is not None and bool(cancel_callback(curr)):
#             return None
        
#         curr_best = dp[curr.path.ijk]
#         if (curr_best is not None) and (curr.cost > curr_best.cost):
#             continue

#         # Termination condition
#         # Near-end option: 
#         # resolve direct connection (Reeds–Shepp), use it as the final segment if successful
#         cx, cy, cyaw = _end_pose(curr)
#         if _reached_goal(cx, cy, cyaw):
#             rs_last = _generate_rspath(curr)
#             if rs_last is not None:
#                 return _reconstruct_path(rs_last)
#             return _reconstruct_path(curr)
        
#         # When close to the goal, first try Reeds–Shepp direct connection and push into the heap to speed up
#         ex, ey, _ = _end_pose(curr)
#         if np.hypot(ex - goal[0], ey - goal[1]) <= REEDS_SHEPP_MAX_DISTANCE:
#             if (rsnode := _generate_rspath(curr)) is not None:
#                 if RETURN_RS_PATH_IMMEDIATELY:
#                     return _reconstruct_path(rsnode)
#                 heapq.heappush(pq, rsnode)

#         for child in _generate_neighbors(curr):
#             if dp[child.path.ijk] is None or child.cost < dp[child.path.ijk].cost:
#                 dp[child.path.ijk] = child
#                 heapq.heappush(pq, child)
        
#     return None


import heapq
from collections.abc import Callable, Generator
from itertools import islice, product
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from rsplan import Path as RSPath
from rsplan.planner import _solve_path as solve_rspath

from ..constants import *
from ..modeling.car import Car
from ..modeling.obstacles import ObstacleGrid, Obstacles
from ..utils.SupportsBool import SupportsBool
from ..utils.wrap_angle import wrap_angle

XY_GRID_RESOLUTION = 1.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_DISTANCE = XY_GRID_RESOLUTION * 1.5  # [m] path interpolate distance
NUM_STEER_COMMANDS = 10  # number of steer command

REEDS_SHEPP_MAX_DISTANCE = 10.0  # maximum distance to use Reeds-Shepp path

SWITCH_DIRECTION_COST = 25.0  # switch direction cost
BACKWARDS_COST = 4.0  # backward movement cost
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


MOVEMENTS = tuple((di, dj, np.sqrt(di**2 + dj**2)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)


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
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not grid.grid[ni, nj] and d + cost < dist[ni, nj]:
                dist[ni, nj] = d + cost
                heapq.heappush(pq, (d + cost, (ni, nj)))
    return ObstacleGrid(grid.minx, grid.maxx, grid.miny, grid.maxy, grid.resolution, dist)


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


def hybrid_a_star(
    start: npt.NDArray[np.floating[Any]],
    goal: npt.NDArray[np.floating[Any]],
    obstacles: Obstacles,
    cancel_callback: Optional[Callable[[Node], SupportsBool]] = None,
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
        XY_GRID_RESOLUTION, min(Car.COLLISION_LENGTH, Car.COLLISION_WIDTH) / 2
    )

    # Precompute the distance to the goal from each grid cell, where the distance will be used as a heuristic
    heuristic_grid = _distance_heuristic(obstacle_grid, goal[:2])
    N, M = heuristic_grid.grid.shape
    K = int(2 * np.pi / YAW_GRID_RESOLUTION)

    # Used to record the path and cost for each grid cell at A* search stage,
    # where dp[y][x][yaw] is the Node object for the grid cell (x, y) with yaw angle yaw
    dp = np.full((N, M, K), None, dtype=Node)


    def calc_ijk(x: float, y: float, yaw: float) -> tuple[int, int, int]:
        "[x, y, yaw] -> [i, j, k] for dp"
        i, j = heuristic_grid.calc_index([x, y])
        k = int(wrap_angle(yaw, zero_to_2pi=True) // YAW_GRID_RESOLUTION)
        return i, j, k

    def generate_neighbour(cur: Node, direction: int, steer: float) -> Optional[Node]:
        "Generate a neighbour node of the current node, given the direction and steer angle"

        # Simulate the car movement for MOTION_DISTANCE, with a interval of MOTION_RESOLUTION,
        # check if the car will collide with the obstacles during the movement
        car = Car(*cur.path.trajectory[-1, :3], velocity=float(direction), steer=steer)
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
            SWITCH_DIRECTION_COST if cur.path.direction != 0 and direction != cur.path.direction else 0.0
        )
        steer_change_cost = STEER_CHANGE_COST * abs(steer - cur.path.steer)
        steer_cost = STEER_COST * abs(steer) * MOTION_DISTANCE
        cost = cur.cost + distance_cost + switch_direction_cost + steer_change_cost + steer_cost

        # Calculate the heuristic cost from this neighbour node to the goal
        h_dist_cost = H_DIST_COST * heuristic_grid.grid[i, j]
        h_yaw_cost = H_YAW_COST * abs(wrap_angle(goal[2] - car.yaw))
        h_cost = h_dist_cost + h_yaw_cost

        return Node(SimplePath((i, j, k), np.array(trajectory), direction, steer), cost, h_cost, cur)

    def generate_neighbours(cur: Node) -> Generator[Node, None, None]:
        "Generate all possible neighbours of the current node"
        nonlocal start_collided
        for direction, steer in product([1, -1], STEER_COMMANDS):
            if (res := generate_neighbour(cur, direction, steer)) is not None:
                yield res
        start_collided = False

    def generate_rspath(node: Node) -> Optional[Node]:
        """
        Try to generate a Path from the current node directly to the goal using Reeds-Shepp curves,
        which will speed up the search process when the node is close to the goal and heuristics
        are not enough to guide the search.
        """

        def check(path: RSPath) -> bool:
            for x, y, yaw in zip(*path.coordinates_tuple()):
                if Car(x, y, yaw).check_collision(obstacles):
                    return False
            return True

        def calc_rspath_cost(path: RSPath) -> float:
            """
            Same logic to calculate the cost of a path in the `generate_neighbour` above, except that:

            1. the heuristic cost is 0, since the Reeds-Shepp path is directly from the current node to the goal.
            2. the cost of the path is calculated as the sum of the cost of each segment in the path.
            """
            last_direction = node.path.direction
            last_steer = node.path.steer

            distance_cost = 0.0
            switch_direction_cost = 0.0
            steer_change_cost = 0.0
            steer_cost = 0.0
            for segment in path.segments:
                length = abs(segment.length)
                distance_cost += length if segment.direction == 1 else length * BACKWARDS_COST
                if last_direction != 0 and segment.direction != last_direction:
                    switch_direction_cost += SWITCH_DIRECTION_COST
                last_direction = segment.direction
                steer = {"left": Car.TARGET_MAX_STEER, "right": -Car.TARGET_MAX_STEER, "straight": 0.0}[segment.type]
                steer_change_cost += STEER_CHANGE_COST * abs(steer - last_steer)
                last_steer = steer
                steer_cost += STEER_COST * abs(steer) * length
            return distance_cost + switch_direction_cost + steer_change_cost + steer_cost

        # generate all possible Reeds-Shepp pathes
        pathes = solve_rspath(
            tuple(node.path.trajectory[-1, :3]), tuple(goal), Car.TARGET_MIN_TURNING_RADIUS, MOTION_RESOLUTION
        )

        # filter out the pathes that collide with the obstacles
        pathes = filter(check, pathes)

        # calculate the cost of each path
        pathes = ((path, calc_rspath_cost(path)) for path in pathes)

        # return the path with the minimum cost
        if (ret := min(pathes, key=lambda t: t[1], default=None)) is None:
            return None
        path, cost = ret
        return Node(path, node.cost + cost, 0.0, node)
    
    def _ensure_dir_sign(traj: np.ndarray) -> np.ndarray:
        """
        输入 [N,3] 或 [N,4]（x,y,yaw[,dir]），输出 [N,4]，把 dir 统一成 ±1，绝不为 0。
        规则：用相邻位移在朝向上的投影判断是前进(+1)还是倒车(-1)；无法判断时沿用前一个；第一个默认 +1。
        """
        assert traj.ndim == 2 and traj.shape[1] in (3, 4)
        if traj.shape[1] == 3:
            traj = np.hstack([traj, np.zeros((traj.shape[0], 1), float)])

        dirc = traj[:, 3].astype(float).copy()
        N = traj.shape[0]
        for k in range(1, N):
            if dirc[k] == 0:
                dx = traj[k, 0] - traj[k-1, 0]
                dy = traj[k, 1] - traj[k-1, 1]
                yaw = traj[k, 2]
                proj = dx * math.cos(yaw) + dy * math.sin(yaw)
                if proj > 1e-6:
                    dirc[k] = 1.0
                elif proj < -1e-6:
                    dirc[k] = -1.0
                else:
                    dirc[k] = dirc[k-1] if dirc[k-1] != 0 else 1.0
        if dirc[0] == 0:
            # 如果第一个还没定，就看第一个非零，否则默认前进
            nz = np.flatnonzero(dirc)
            dirc[0] = dirc[nz[0]] if nz.size else 1.0

        traj[:, 3] = np.sign(dirc)
        return traj

    def traceback_path(node: Node) -> npt.NDArray[np.floating[Any]]:
        segs = []
        cur = node
        while cur is not None:
            if isinstance(cur.path, RSPath):
                xs, ys, yaws = cur.path.coordinates_tuple()  
                pts = np.stack([xs, ys, yaws], axis=1)
                
                d = cur.parent.path.direction if cur.parent and not isinstance(cur.parent.path, RSPath) else 1
                dircol = np.full((pts.shape[0], 1), d, float)
                segs.append(np.hstack([pts, dircol]))
            else:
                traj = cur.path.trajectory  # [N,3]
                dircol = np.full((traj.shape[0], 1), cur.path.direction, float)
                segs.append(np.hstack([traj, dircol]))
            cur = cur.parent
        segs.reverse()
        return _ensure_dir_sign(np.vstack(segs))

    if start_is_point:
        start_ijk = calc_ijk(*start)
        start_path = SimplePath(start_ijk, np.array([start]), 0, 0.0)
    else:
        xy = start[:, :2]
        mask = (xy[:-1] != xy[1:]).any(axis=1)  # remove consecutive identical points
        start = start[np.concatenate(([True], mask))]
        start_ijk = calc_ijk(*start[-1, :3])
        start[:, 3] = np.sign(start[0, 3])
        steer = 0.0
        if start.shape[0] >= 2 and (l := np.linalg.norm(start[-1, :2] - start[-2, :2])):
            steer = np.arctan(Car.WHEEL_BASE * (start[-1, 2] - start[-2, 2]) / l)
        start_path = SimplePath(start_ijk, start, start[0, 3], steer)
    start_node = Node(start_path, 0.0, H_DIST_COST * heuristic_grid.grid[start_ijk[:2]], None)

    dp[start_ijk] = start_node
    pq = [start_node]
    while pq:  # A* search (Similar to Dijkstra's algorithm, but with a heuristic cost added)
        cur = heapq.heappop(pq)
        if isinstance(cur.path, RSPath):
            if cancel_callback is not None and cancel_callback(cur):
                return None  # canceled
            return traceback_path(cur)

        if cur.cost > dp[cur.path.ijk].cost:
            continue

        if cancel_callback is not None and cancel_callback(cur):
            return None  # canceled

        if np.linalg.norm(cur.path.trajectory[-1, :2] - goal[:2]) <= REEDS_SHEPP_MAX_DISTANCE:
            if (rsnode := generate_rspath(cur)) is not None:
                if RETURN_RS_PATH_IMMEDIATELY:
                    return traceback_path(rsnode)
                heapq.heappush(pq, rsnode)

        for neighbour in generate_neighbours(cur):
            if dp[neighbour.path.ijk] is None or neighbour.cost < dp[neighbour.path.ijk].cost:
                dp[neighbour.path.ijk] = neighbour
                heapq.heappush(pq, neighbour)
    return None
