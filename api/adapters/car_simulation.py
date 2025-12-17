import threading
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import scipy.interpolate

import sys
from pathlib import Path as _Path
_project_root = _Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AutonomousVehicle.modeling.car import Car
from api.event_types import *
from .event_bus import EventBus

class CarSimulationAdapter:
    """
    Car Simulation Adapter

    Features:
    - Periodically updates the vehicle state (simulation)
    - Periodically publishes vehicle state (via event bus)
    - Receives and applies control sequences to the vehicle
    """

    def __init__(
        self,
        event_bus: EventBus,
        delta_time_s: float,
        simulation_interval_s: float,
        publish_interval_s: float,
    ) -> None:
        """
        Initialize the car simulation adapter
        
        Args:
            event_bus: Event bus instance
            delta_time_s: Simulation time step (seconds)
            simulation_interval_s: Simulation update interval (seconds)
            publish_interval_s: State publish interval (seconds)
        """
        self._event_bus = event_bus
        self._real_state: Optional[Car] = None
        self._control_tck: Optional[tuple[npt.NDArray[np.floating[Any]], ...]] = None
        self._control_u: Optional[npt.NDArray[np.floating[Any]]] = None
        self._delta_time_s = delta_time_s
        self._timestamp_s = 0.0
        self._stopped = True

        # timer
        self._simulation_interval = simulation_interval_s
        self._publish_interval = publish_interval_s
        self._simulation_timer: Optional[threading.Timer] = None
        self._publish_timer: Optional[threading.Timer] = None
        
        # lock
        self._timer_lock = threading.RLock()
        self._running = False

    def _simulate(self) -> None:
        """Execute a simulation once
        
        NOTE: This method modifies shared state and should be called
        while holding self._timer_lock for thread safety.
        """
        if self._real_state is None:
            return 
        self._timestamp_s += self._delta_time_s

        if self._control_tck is None:
            self._real_state.update(self._delta_time_s)
            return
        
        # update car state
        t = np.clip(self._timestamp_s, self._control_u[0], self._control_u[-1])
        velocity, steer = scipy.interpolate.splev(t, self._control_tck)
        self._real_state.update_with_control(velocity, steer, self._delta_time_s)

    def _publish_state(self, timestamp: float, real_state: Car) -> None:
        """Publish a car state to event_bus
    
        Args:
            timestamp: Simulation timestamp
            real_state: Car state object (already copied, guaranteed to be not None)
        """
        self._event_bus.emit(MEASURED_STATE, timestamp, real_state)
    
    def _schedule_simulation(self) -> None:
        """
        Schedule the next simulation update.
        Use threading.Timer for periodic invocation.
        """
        with self._timer_lock:
            if not self._running:
                return
            
            # simulate
            self._simulate()
            
            # schedule next simulation
            self._simulation_timer = threading.Timer(
                self._simulation_interval,
                self._schedule_simulation
            )
            self._simulation_timer.daemon = True
            self._simulation_timer.start()

    
    def _schedule_publish(self) -> None:
        """
        Schedule next status publishment
        """
        with self._timer_lock:
            if not self._running:
                return
            
            if self._real_state is None:
                return
            timestamp = self._timestamp_s
            car_copy = self._real_state.copy()
        
        # publish
        self._publish_state(timestamp, car_copy)
        
        with self._timer_lock:
            if not self._running:
                return
            # schedule the next publish
            self._publish_timer = threading.Timer(
                self._publish_interval,
                self._schedule_publish
            )
            self._publish_timer.daemon = True
            self._publish_timer.start()
    
    def start(self) -> None:
        """
        Start simulation and publish status
        """

        with self._timer_lock:
            if self._running:
                return 
            
            self._running = True
            self._schedule_simulation()
            self._schedule_publish()
    
    def stop(self) -> None:
        """
        Stop simulation
        """
        
        with self._timer_lock:
            self._running = False

            if self._simulation_timer:
                self._simulation_timer.cancel()
                self._simulation_timer = None
            
            if self._publish_timer:
                self._publish_timer.cancel()
                self._publish_timer = None
            
            if self._real_state is not None:
                self._real_state.velocity = 0.0
                self._real_state.steer = 0.0
            
            self._control_tck = None
            self._stopped = True

    def set_control_sequence(self, control_sequence: npt.NDArray[Any]) -> None:
        """
        Set the control sequence.
        
        Args:
            control_sequence: Control sequence, shape (N, 3). Each row is [timestamp, velocity, steer].
        """
        with self._timer_lock:
            if self._stopped:
                return
            timestamps, controls = control_sequence[:, 0], control_sequence[:, 1:]
            self._control_tck, self._control_u = scipy.interpolate.splprep(
                controls.T,
                s=0,
                k=1,
                u=timestamps
            )
    
    def set_state(self, state: Car) -> None:
        """
        Set the initial state of the car.
        
        Args:
            state: Car state object
        """
        with self._timer_lock:
            self._real_state = state.copy()
    
    def resume(self) -> None:
        """
        Resume simulation (recover from stopped state)
        """
        with self._timer_lock:
            self._stopped = False
    
    def get_state(self) -> Optional[tuple[float, Car]]:
        """
        Get the current state (for HTTP query)
        
        Returns:
            (timestamp, car_state) or None
        """
        with self._timer_lock:
            if self._real_state is None:
                return None
            return (self._timestamp_s, self._real_state.copy())
