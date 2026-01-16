import json
from typing import Any, Optional
import numpy as np
import numpy.typing as npt

from AutonomousVehicle.modeling.Car import Car

def serialize_car(car: Car) -> dict:
    """
    Serialize a Car object to a dictionary.

    Args:
        car: Car object

    Returns:
        Dictionary containing the car state

    Example:
        car = Car(x=5.0, y=3.0, yaw=0.0, velocity=10.0, steer=0.1)
        result = serialize_car(car)
        # {'x': 5.0, 'y': 3.0, 'yaw': 0.0, 'velocity': 10.0, 'steer': 0.1}
    """
    return {
        'x': float(car.x),
        'y': float(car.y),
        'yaw': float(car.yaw),
        'velocity': float(car.velocity),
        'steer': float(car.steer)
    }

def serialize_trajectory(trajectory: npt.NDArray[np.floating[Any]]) -> list:
    """
    Serialize a numpy trajectory array to a list.

    The trajectory format is usually [N, 3] or [N, 4], containing [x, y, yaw, ...]

    Args:
        trajectory: numpy array, shape [N, M], M >= 2

    Returns:
        Nested list, each element is a point

    Example:
        traj = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
        result = serialize_trajectory(traj)
        # [[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]
    """
    if trajectory is None or trajectory.size == 0:
        return []

    return trajectory.tolist()

def serialize_obstacles(obstacles: Optional[npt.NDArray[np.floating[Any]]]) -> list:
    """
    Serialize a numpy obstacle coordinate array to a list.

    Args:
        obstacles: numpy array, shape [N, 2], containing [x, y] coordinates

    Returns:
        Nested list, each element is [x, y]

    Example:
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = serialize_obstacles(obs)
        # [[1.0, 2.0], [3.0, 4.0]]
    """
    if obstacles is None or obstacles.size == 0:
        return []
    
    return obstacles.tolist()

def serialize_numpy_array(arr: Optional[npt.NDArray[np.floating[Any]]]) -> Optional[list]:
    """
    General numpy array serialization function

    Args:
        arr: numpy array (any dimension)

    Returns:
        Nested list
    """
    if arr is None:
        return None
    
    if isinstance(arr, np.ndarray):
        return arr.tolist()

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types.

    Usage:
        json.dumps(data, cls=NumpyEncoder)

    Note: Flask's jsonify can already handle numpy types; this is mostly for other scenarios.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Car):
            return serialize_car(obj)
        return super().default(obj)