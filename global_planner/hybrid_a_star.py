import math, heapq
import numpy as np
from modeling.obstacles import ObstacleGrid


H_COLLISION_COST = 1e4  # collision cost when calculating heuristic

MOVEMENTS = tuple((di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)

import heapq, math
import numpy as np

def _distance_field(grid: ObstacleGrid, goal_xy: tuple[float, float]) -> np.ndarray:
    H, W = grid.grid.shape
    INF = float('inf')
    dist = np.full((H, W), H_COLLISION_COST)

    gi, gj = grid.calc_index(goal_xy)
    if not (0 <= gi < H and 0 <= gj < W):   
        return dist
    if grid.grid[gi, gj]:                   
        return dist

    dist[gi, gj] = 0.0
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, (gi, gj))]

    while pq:
        distance, (i, j) = heapq.heappop(pq)
        if distance > dist[i, j]:
            continue
        for di, dj, step_cost in MOVEMENTS:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not grid.grid[ni, nj]:
                nd = distance + step_cost
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(pq, (nd, (ni, nj)))
    return dist
