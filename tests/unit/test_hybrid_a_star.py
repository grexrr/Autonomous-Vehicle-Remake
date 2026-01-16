import numpy as np
import pytest

from AutonomousVehicle.global_planner.hybrid_a_star import hybrid_a_star
from AutonomousVehicle.modeling.Car import Car
from AutonomousVehicle.modeling.Obstacles import Obstacles


class TestHybridAStar:
    @staticmethod
    def _bounding_obstacles(*, minx: float, maxx: float, miny: float, maxy: float) -> Obstacles:
        """
        Case: Hybrid A* internally downsamples obstacles into a grid whose bounds come from obstacle coordinates.
        So tests must provide obstacle points that *cover* start/goal, even if obstacles are far away.
        """
        coords = np.array(
            [
                [minx, miny],
                [minx, maxy],
                [maxx, miny],
                [maxx, maxy],
            ],
            dtype=float,
        )
        return Obstacles(coords)

    def test_finds_path_in_open_space(self):
        """
        Case: start -> goal with no blocking obstacles should produce a trajectory.
        Expectation: returns a non-empty Nx4 array [x, y, yaw, direction].
        """
        start = np.array([0.0, 0.0, 0.0], dtype=float)
        goal = np.array([10.0, 0.0, 0.0], dtype=float)
        obstacles = self._bounding_obstacles(minx=-10.0, maxx=20.0, miny=-10.0, maxy=10.0)

        traj = hybrid_a_star(start, goal, obstacles)

        assert traj is not None
        assert isinstance(traj, np.ndarray)
        assert traj.ndim == 2 and traj.shape[1] == 4
        assert traj.shape[0] >= 2

        # Start/end should be near requested endpoints (loose tolerance: planner is discretized).
        assert np.allclose(traj[0, :2], start[:2], atol=1e-6)
        assert np.linalg.norm(traj[-1, :2] - goal[:2]) <= 2.0

        # Direction should be either forward(1) or backward(-1)
        assert set(np.unique(traj[:, 3])).issubset({-1.0, 1.0})

    def test_returns_none_if_goal_in_collision(self):
        """
        Case: goal pose is already in collision -> no valid plan.
        Expectation: returns None.
        """
        start = np.array([0.0, 0.0, 0.0], dtype=float)
        goal = np.array([10.0, 0.0, 0.0], dtype=float)

        # Put an obstacle point at the goal car's center, guaranteeing collision.
        goal_car = Car(*goal)
        c, s = np.cos(goal_car.yaw), np.sin(goal_car.yaw)
        center_x = goal_car.x + goal_car.BACK_TO_CENTER * c
        center_y = goal_car.y + goal_car.BACK_TO_CENTER * s
        collision_point = np.array([[center_x, center_y]], dtype=float)

        bounds = np.array([[-10.0, -10.0], [-10.0, 10.0], [20.0, -10.0], [20.0, 10.0]], dtype=float)
        obstacles = Obstacles(np.vstack([bounds, collision_point]))

        traj = hybrid_a_star(start, goal, obstacles)
        assert traj is None

    def test_can_be_cancelled(self):
        """
        Case: user cancels planning.
        Expectation: returns None quickly once cancel_callback says True.
        """
        start = np.array([0.0, 0.0, 0.0], dtype=float)
        goal = np.array([15.0, 5.0, np.pi / 4], dtype=float)
        obstacles = self._bounding_obstacles(minx=-10.0, maxx=25.0, miny=-10.0, maxy=15.0)

        traj = hybrid_a_star(start, goal, obstacles, cancel_callback=lambda _node: True)
        assert traj is None

    def test_accepts_start_as_trajectory_and_keeps_prefix(self):
        """
        Case: replanning from an existing trajectory (Nx4).
        Expectation: returned trajectory begins with the provided start trajectory (as a prefix).
        """
        # A short prior trajectory moving forward along x.
        start_traj = np.array(
            [
                [0.0, 0.0, 0.0, 2.0],
                [1.0, 0.0, 0.0, 2.0],
                [2.0, 0.0, 0.0, 2.0],
            ],
            dtype=float,
        )
        goal = np.array([10.0, 0.0, 0.0], dtype=float)
        obstacles = self._bounding_obstacles(minx=-10.0, maxx=20.0, miny=-10.0, maxy=10.0)

        traj = hybrid_a_star(start_traj, goal, obstacles)

        assert traj is not None
        assert traj.ndim == 2 and traj.shape[1] == 4
        assert traj.shape[0] >= start_traj.shape[0]

        # Prefix should match the provided start trajectory (direction column becomes sign(velocity)).
        assert np.allclose(traj[:3, :3], start_traj[:, :3], atol=1e-6)
        assert np.all(traj[:3, 3] == 1.0)

    def test_validates_input_shapes(self):
        """
        Case: invalid inputs should fail fast (better UX than silent wrong planning).
        """
        obstacles = self._bounding_obstacles(minx=-10.0, maxx=20.0, miny=-10.0, maxy=10.0)
        goal = np.array([10.0, 0.0, 0.0], dtype=float)

        with pytest.raises(AssertionError):
            hybrid_a_star(np.array([0.0, 0.0], dtype=float), goal, obstacles)

        with pytest.raises(AssertionError):
            hybrid_a_star(np.array([0.0, 0.0, 0.0], dtype=float), np.array([10.0, 0.0], dtype=float), obstacles)


