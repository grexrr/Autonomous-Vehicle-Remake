from typing import Any

import numpy as np
import numpy.typing as npt

def wrap_angle(theta: float, zero_to_2pi: bool = False):
    """
    Wrap angle to [-π, π] or [0, 2π] range
    
    Args:
        theta: Input angle in radians
        zero_to_2pi: If True, wrap to [0, 2π], otherwise wrap to [-π, π]
    
    Returns:
        Wrapped angle in the specified range
    """
    if zero_to_2pi:
        # Wrap to [0, 2π]
        return theta % (2.0 * np.pi)
    else:
        # Wrap to [-π, π] (original behavior)
        return (theta + np.pi) % (2.0 * np.pi) - np.pi
    

def smooth_yaw(yaws: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    "Make the yaws along a trajectory continuous, preventing sudden changes of -2pi -> 2pi"
    diff = np.diff(yaws, prepend=0)
    diff = wrap_angle(diff)
    return np.cumsum(diff)