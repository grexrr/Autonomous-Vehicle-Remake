# Based on a linearized bicycle model (state includes x, y, v, yaw, control includes acceleration a, steering rate or angle), 
# solving quadratic programming (QP) at each time step.
# The cost includes:
# R: Control usage cost (suppress sudden throttle/steering changes)
# R_D: Control increment cost (suppress jitter)
# Q / Q_F: State error (trajectory tracking, yaw alignment, speed convergence).

# Additionally, hard constraints on steering angle/speed/acceleration are included. 
# This allows for smooth and robust tracking of the reference trajectory provided by Hybrid A*.

from typing import NamedTuple, Optional, Any

import cvxpy
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.optimize

from ..modeling.Car import Car



HORIZON_LENGTH = 5  # simulate count
MIN_HORIZON_DISTANCE = 0.3  # [m]


MAX_ITER = 5
DU_TH = 0.1  # iteration finish param

NX = 4  # [x, y, v, yaw]
NU = 2  # [accel, steer]

DIRECTION_CHANGE_DIST = 0.1  # [m] distance to change the direction


def _predict_motion(state: Car, controls: npt.NDArray[np.floating[Any]], dt: float) -> npt.NDArray[np.floating[Any]]:
    return

def _linear_mpc_control(
    xref: npt.NDArray[np.floating[Any]], xbar: npt.NDArray[np.floating[Any]], last_steer: float, dt: float
) -> Optional[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    return

def _get_curvature(
    tck: tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], int], u: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    return


class MPCResult(NamedTuple):
    controls: npt.NDArray[np.floating[Any]]  # [[accel, steer]], target output controls
    states: npt.NDArray[np.floating[Any]]  # [[x, y, v, yaw]], predicted states
    ref_states: npt.NDArray[np.floating[Any]]  # [[x, y, v, yaw]], reference states on the trajectory
    brake_trajectory: npt.NDArray[np.floating[Any]]  # [[x, y, v, yaw]], trajectory when braking

class ModelPredictiveControl:
    def __init__(self, ref_trajectory: npt.NDArray[np.floating[Any]]) -> None:
        assert ref_trajectory.shape[1] == 4, "Reference trajectory have [[x, y, yaw, direction], ...]"
        assert (ref_trajectory[:, 3] != 0).all(), "the direction on each point of the trajectory should not be zero"


        # calculate the ticks of the reference trajectory
        dists = np.linalg.norm(ref_trajectory[1:, :2] - ref_trajectory[:-1, :2], axis=1)
        u = np.concatenate(([0], np.cumsum(dists))) # indicates the "progress parameter" along the trajectory

        # record the points where the direction of the vehicle changes
        self._direction_changing_us = u[ref_trajectory[:, 2] == 0.0][:-1]

        # Spline fitting is performed on the four trajectories [x, y, v, yaw] simultaneously to enable smooth interpolation of each attribute (position, speed, heading) along the trajectory using the parameter u,
        tck, _ = scipy.interpolate.splprep(ref_trajectory.T[:2], s=0, k=2, u=u)
        
        # calculate the curvature of the reference trajectory
        curvature = _get_curvature(tck, u)
        
     

        self._tck, _ = scipy.interpolate.splprep(ref_trajectory.T, s=0, k=1, u=u)    
        self._cur_u = 0.0  
        self._u_limit = u[-1]    # max u (the end) of ref_trajectory

        self._brake_trajectory = np.zeros((6,4)) 
        self._brake = self._braked = False
        return
    
    def _find_nearist_point(self, state: Car) -> None:
        "find the nearist point on the reference trajectory to the given state"
        return
    
    def _find_xref(self, state: Car, dt:float) -> npt.NDArray[np.floating[Any]]:
        "find the closest point in the reference trajectory, and interpolate the reference trajectory within a horizon"

        while True:
            self._find_nearist_point(state)
            
            # =========== 1. interpolate the reference trajectory  =========== 

            # Reference trajectory's reference speed v_ref(u) at position cur_u
            v = np.sign(
                scipy.interpolate.splev(
                    self._cur_u,
                    self._tck
                )[2]) * state.velocity
            
            # When the car is fast: "look" further ahead to avoid abrupt control
            # When the car is slow: "look" at least MIN_HORIZON_DISTANCE ahead to avoid jittering 
            length = max(
                MIN_HORIZON_DISTANCE,
                max(0, v) * dt * HORIZON_LENGTH
            )
            # sample H + 1 of u
            ref_u = np.linspace(self._cur_u, self._cur_u + length, HORIZON_LENGTH + 1)
            ref_u = np.clip(ref_u, a_min=None, a_max=self._u_limit)

            # read u_state from the spline
            # returns 6 * 4 ([x0, y0, v0, yaw0])
            xref = np.array(scipy.interpolate.splev(ref_u, self._tck)).T

            # =========== 2. Check whether the horizon crosses the "next gear shifting point"  ===========
            # self._direction_changing_us[i - 1] <= self._cur_u < self._direction_changing_us[i]
            i = np.searchsorted(self._direction_changing_us, self._cur_u, side="right")
            changing_point = self._direction_changing_us[i] if i < len(self._direction_changing_us) else np.inf

            if ref_u[-1] >= changing_point:  # if the reference trajectory contains a direction change

                # if the direction change happens immediately after the first point, we discard the first point
                # and start to track the trajectory from the direction changing point
                if self._cur_u + DIRECTION_CHANGE_DIST >= changing_point:
                    self._cur_u = changing_point
                    continue

                # otherwise the direction changing point is set to have v = 0 and the vehicle should stop at this point
                i = np.searchsorted()
                
    
    def update(self, state: Car, dt: float) -> MPCResult:
        xref = self._find_xref(state, dt)
        
        # Align the yaw of the vehicle with the reference trajectory, to facilitate the calculation of
        # the yaw difference between the current state and the reference trajectory.
        state = state.copy()
        state.align_yaw(xref[0, 3])

        # iteratively solve the linearized problem
        controls, states = np.zeros((HORIZON_LENGTH, NU)), np.zeros((HORIZON_LENGTH + 1), NX)

        for _ in range(MAX_ITER):
            # predict a rollout based on current state, control
            xbar = _predict_motion(state, controls, dt)
            pre_controls = controls.copy()

            # solve the optimized control near xbar
            res = _linear_mpc_control(xref.T, xbar.T, state.steer, dt)
            if res is None:
                break
            controls, states = res[0].T, res[1].T
            du = np.linalg.norm(controls - pre_controls)
            # convergence threshold
            if du < DU_TH:
                break
        else:
            print("Warning: Cannot converge mpc!")
        
        return MPCResult(
            controls=controls, 
            states=states[:, [0, 1, 3, 2]], 
            ref_states=xref[:, [0, 1, 3, 2]], 
            brake_trajectory=self._brake_trajectory[:, [0, 1, 3, 2]]
        )