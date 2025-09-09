import numpy as np
from dataclasses import dataclass

@dataclass
class ObstacleGrid:
    minx: float
    maxx: float
    minx: float
    miny: float
    resolution: float
    grid: np.ndarray

class Obstacles:
    def __init__(self):
        self._coordinates = []  # list of (x, y)

    def downsampling_to_grid(self, resolution: float, radius: float) -> ObstacleGrid:
        half_res = resolution / 2

        minx, maxx = np.min(self._coordinates[:, 0]) - half_res, np.max(self._coordinates[:, 0]) + half_res
        miny, maxy = np.min(self._coordinates[:, 1]) - half_res, np.max(self._coordinates[:, 1]) + half_res

        x_count, y_count = round((maxx - minx)/resolution), round((maxy - miny)/resolution)
        maxx, maxy = minx + x_count * resolution, miny + y_count * resolution


        x_centers = np.arange(minx + half_res, maxx, resolution)
        y_centers = np.arange(miny + half_res, maxy,  resolution)
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

        points = np.array([X, Y]).T.reshape(-1, 2)
        
        dist, _ = self.kd_tree.query(points, k=1, distance_upper_bound=radius + resolution)
        grid = (dist <= radius).reshape(y_count, x_count)

        return ObstacleGrid(minx, maxx, miny, maxy, resolution, grid)
