import numpy as np

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