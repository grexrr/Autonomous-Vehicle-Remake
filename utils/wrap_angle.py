import numpy as np

def wrap_angle(theta:float):
    theta = (theta + np.pi) % (2.0*np.pi) - np.pi
    return theta