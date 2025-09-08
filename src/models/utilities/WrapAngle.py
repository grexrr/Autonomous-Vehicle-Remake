import numpy as np

def wrap_to_pi(theta:float):
    theta = (theta + np.pi) % (2.0*np.pi) - np.pi
    return theta