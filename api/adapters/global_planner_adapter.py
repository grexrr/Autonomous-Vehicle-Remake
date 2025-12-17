
from enum import Enum, auto
from multiprocessing.connection import Connection
from typing import Any, Optional, List

import numpy as np
import numpy.typing as npt

import sys
from pathlib import Path
# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AutonomousVehicle.global_planner.hybrid_a_star import Node, hybrid_a_star
from AutonomousVehicle.modeling.car import Car
from AutonomousVehicle.modeling.obstacles import Obstacles, ObstacleGrid
from api.event_types import GLOBAL_PLANNER_DISPLAY_SEGMENTS, GLOBAL_PLANNER_TRAJECTORY, GLOBAL_PLANNER_FINISHED

from .process_adapter import ProcessAdapter
from .event_bus import EventBus


# ========== Message Type Enums ==========
# These enums define the communication protocol between the parent process and the child process.

class _ParentMsgType(Enum):
    """Message types sent from the parent process to the child process"""
    PLAN = auto()      # Request to plan a path
    CANCEL = auto()    # Cancel the current planning

class _WorkerMsgType(Enum):
    """Message types sent from the child process to the parent process"""
    DISPLAY_SEGMENTS = auto()  # Intermediate visualization data
    TRAJECTORY = auto()        # Final trajectory result

# ========== Worker Process Function ==========
# This function runs in the child process and performs the actual path planning algorithm.
# Note: This function is completely independent of Qt, so it can remain unchanged!

def _worker_process(pipe: Connection, segment_collection_size: int) -> None:
    """
    Worker process function that runs in a separate process
    
    This function:
    1. Receives planning requests from parent process
    2. Runs hybrid_a_star algorithm
    3. Sends intermediate results (display_segments) and final trajectory back
    
    Args:
        pipe: Connection to parent process
        segment_collection_size: Number of segments to collect before sending
    """
    while True:
        match pipe.recv():
            case _ParentMsgType.CANCEL:
                continue
            case _ParentMsgType.PLAN, start, goal, obstacles: 
            # self._worker.send((_ParentMsgType.PLAN, start, goal, obstacles))
                if pipe.poll():
                    continue
                
                display_segments: List[npt.NDArray[np.floating[Any]]] = []

                def callback(node: Node) -> bool:
                    """
                    Callback function called during path planning
                    Returns True to stop planning, False to continue
                    """

                    display_segments.append(node.get_plot_trajectory())
                    
                    if len(display_segments) < segment_collection_size:
                        return False
                    
                    # if there's new mission, stop
                    if pipe.poll():
                        return True

                    # sent intermediate result to main process
                    pipe.send(_WorkerMsgType.DISPLAY_SEGMENTS, display_segments)
                    display_segments.clear()
                    return False
                
                # the actual algorithm
                traj = hybrid_a_star(start, goal, obstacles, callback)

                if not pipe.poll:
                    pipe.send(_WorkerMsgType.TRAJECTORY, traj)

class GlobalPlannerAdapter:
    """
    Adapter for GlobalPlanner
    
    This adapter:
    1. Uses ProcessAdapter to run path planning in a separate process
    2. Publishes events via EventBus instead of Qt Signals
    3. Maintains the same interface as GlobalPlannerNode
    
    Events published:
    - GLOBAL_PLANNER_FINISHED: () - Planning completed
    - GLOBAL_PLANNER_TRAJECTORY: (trajectory: np.ndarray) - Final trajectory
    - GLOBAL_PLANNER_DISPLAY_SEGMENTS: (segments: list) - Intermediate visualization data
    """

    def __init__(
            self,
            event_bus: EventBus,
            segment_collection_size: int
        ) -> None:
        self._event_bus = event_bus
        self._process_adapter = ProcessAdapter(
            event_bus=event_bus,
            event_name='_global_planner_worker_message', # internal event name
            target=_worker_process,
            args=(segment_collection_size,)
        )
        # 订阅内部事件，处理子进程消息
        self._event_bus.subscribe('_global_planner_worker_message', self._handle_worker_message)
    
    def start(self) -> None:
        """Start the working process"""
        self._process_adapter.start()

    def plan(
            self,
            start_state: Car | npt.NDArray[np.floating[Any]],
            goal_state: Car,
            obstacles: Obstacles
        ) -> None:
        """
        Request path planning
        
        Args:
            start_state: Starting state (Car object or numpy array [x, y, yaw])
            goal_state: Goal state (Car object)
            obstacles: Obstacles object
        """
        if isinstance(start_state, Car):
            start = np.array([start_state.x, start_state.y, start_state.yaw])
        else:
            start = start_state
        goal = np.array([goal_state.x, goal_state.y, goal_state.yaw])

        # send planning request to child process
        self._process_adapter.send((_ParentMsgType.PLAN, start, goal, obstacles))
    
    def cancel(self) -> None:
        """Cancel current path planning"""
        self._process_adapter.send(_ParentMsgType.CANCEL)
    
    def _handle_worker_message(self, data) -> None:
        """
        Handle messages from worker process
        
        This method is called when ProcessAdapter receives a message from child process.
        It unpacks the message and publishes appropriate events.
        
        Args:
            data: Message from worker process (tuple of (msg_type, payload))
        """
        match data:
            case _WorkerMsgType.DISPLAY_SEGMENTS, display_segments:
                self._event_bus.emit(GLOBAL_PLANNER_DISPLAY_SEGMENTS, display_segments)
            
            case _WorkerMsgType.TRAJECTORY, trajectory:
                self._event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)

                if trajectory is not None:
                    self._event_bus.emit(GLOBAL_PLANNER_FINISHED)
    
    def stop(self) -> None:
        self._event_bus.unsubscribe('_global_planner_worker_message', self._handle_worker_message)
        self._process_adapter.stop()
    
    def is_alive(self) -> bool:
        return self._process_adapter.is_alive()
                
        
    
