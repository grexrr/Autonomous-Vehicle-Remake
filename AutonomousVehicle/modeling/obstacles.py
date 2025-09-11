# from typing import Any, NamedTuple
# import numpy as np
# import numpy.typing as npt
# from scipy.spatial import KDTree

# class ObstacleGrid(NamedTuple):
#     minx: float
#     maxx: float
#     miny: float
#     maxy: float
#     resolution: float
#     grid: np.ndarray  # shape = [rows(y_count), cols(x_count)]

#     def calc_index(self, xy: tuple[float, float]) -> tuple[int, int]:
#         x, y = xy
#         i = int((y - self.miny) / self.resolution)  # 行（沿 y）
#         j = int((x - self.minx) / self.resolution)  # 列（沿 x）
#         return i, j

# class Obstacles:
#     def __init__(self, coordinates: npt.NDArray[np.floating[Any]]) -> None:
#         assert coordinates.ndim == 2 and coordinates.shape[1] == 2, "Coordinates must be a 2D array of shape (n, 2)"
#         self._coordinates = coordinates
#         self.kd_tree = KDTree(coordinates)
    
#     @property
#     def coordinates(self) -> np.ndarray:
#         return self._coordinates

#     def downsampling_to_grid(self, resolution: float, radius: float) -> ObstacleGrid:
#         half_res = resolution / 2

#         minx, maxx = np.min(self._coordinates[:, 0]) - half_res, np.max(self._coordinates[:, 0]) + half_res
#         miny, maxy = np.min(self._coordinates[:, 1]) - half_res, np.max(self._coordinates[:, 1]) + half_res

#         x_count, y_count = round((maxx - minx)/resolution), round((maxy - miny)/resolution)
#         maxx, maxy = minx + x_count * resolution, miny + y_count * resolution


#         x_centers = np.arange(minx + half_res, maxx, resolution)
#         y_centers = np.arange(miny + half_res, maxy,  resolution)
#         X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

#         points = np.array([X, Y]).T.reshape(-1, 2)
        
#         dist, _ = self.kd_tree.query(points, k=1, distance_upper_bound=radius + resolution)
#         grid = (dist <= radius).reshape(y_count, x_count)

#         return ObstacleGrid(minx, maxx, miny, maxy, resolution, grid)


from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


class ObstacleGrid(NamedTuple):
    minx: float
    maxx: float
    miny: float
    maxy: float
    resolution: float
    grid: np.ndarray

    def calc_index(self, xy: npt.ArrayLike) -> tuple[int, int]:
        x, y = xy
        return int((y - self.miny) / self.resolution), int((x - self.minx) / self.resolution)


class Obstacles:
    def __init__(self, coordinates: npt.NDArray[np.floating[Any]]) -> None:
        assert coordinates.ndim == 2 and coordinates.shape[1] == 2, "Coordinates must be a 2D array of shape (n, 2)"
        self._coordinates = coordinates
        self._kd_tree = KDTree(coordinates)

    @property
    def coordinates(self) -> npt.NDArray[np.floating[Any]]:
        return self._coordinates

    @property
    def kd_tree(self) -> KDTree:
        return self._kd_tree

    def downsampling_to_grid(self, resolution: float, radius: float) -> ObstacleGrid:
        "downsample the obstacles to a grid with a given resolution in meters, and a given collision radius."
        # calculate the range and the size of the grid
        half_res = resolution / 2
        minx, maxx = np.min(self.coordinates[:, 0]) - half_res, np.max(self.coordinates[:, 0]) + half_res
        miny, maxy = np.min(self.coordinates[:, 1]) - half_res, np.max(self.coordinates[:, 1]) + half_res
        x_count, y_count = round((maxx - minx) / resolution), round((maxy - miny) / resolution)
        maxx, maxy = minx + x_count * resolution, miny + y_count * resolution

        # generate a meshgrid of center points of each cell of the grid
        points = np.array(
            np.meshgrid(
                np.arange(minx + half_res, maxx, resolution),
                np.arange(miny + half_res, maxy, resolution),
                indexing="ij",
            )
        ).T.reshape(-1, 2)

        # query the obstacles within the collision radius of each point of the grid, determine whether
        # the center of a cell is within the collision radius of any obstacle
        dist, _ = self.kd_tree.query(points, k=1, distance_upper_bound=radius + resolution)
        grid = (dist <= radius).reshape(y_count, x_count)

        return ObstacleGrid(minx, maxx, miny, maxy, resolution, grid)
