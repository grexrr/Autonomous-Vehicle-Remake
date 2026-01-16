from enum import Enum, auto
from math import pi
from multiprocessing.connection import Connection
from typing import Optional, NamedTuple, Any
import threading
import numpy as np
import numpy.typing as npt

import sys
from pathlib import Path
# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AutonomousVehicle.local_planner.model_predictive_control import ModelPredictiveControl, MPCResult
from AutonomousVehicle.modeling.Car import Car
from api.event_types import MEASURED_STATE, LOCAL_PLANNER_CONTROL_SEQUENCE, LOCAL_PLANNER_TRAJECTORIES
from .event_bus import EventBus
from .process_adapter import ProcessAdapter

class _ParentMsgType(Enum):
    TRAJECTORY = auto()
    STATE = auto()
    BRAKE = auto()
    CANCEL = auto()

# ========== LocalPlanningTrajectories 命名元组 ==========
# 用于封装局部规划的输出数据

class LocalPlanningTrajectories(NamedTuple):
    """局部规划轨迹数据"""
    local_trajectory: npt.NDArray[np.floating[Any]]  # 局部轨迹
    reference_points: npt.NDArray[np.floating[Any]]   # 参考点
    brake_trajectory: npt.NDArray[np.floating[Any]]    # 刹车轨迹


# ========== Worker 进程函数 ==========
# 这个函数在子进程中运行，执行 MPC 控制算法

def _worker_process(pipe: Connection, delta_time_s: float) -> None:
    """
    Worker process function that runs in a separate process
    
    This function:
    1. Receives trajectory, state, brake, and cancel commands
    2. Runs Model Predictive Control (MPC) algorithm
    3. Sends control sequence and local trajectories back
    
    Args:
        pipe: Connection to parent process
        delta_time_s: Time step for MPC (seconds)
    """
    
    mpc: Optional[ModelPredictiveControl] = None
    try:
        while True:
            match pipe.recv():
                case _ParentMsgType.CANCEL:
                    mpc = None
                
                case _ParentMsgType.TRAJECTORY, trajectory:
                    mpc = ModelPredictiveControl(trajectory)
                
                case _ParentMsgType.BRAKE:
                    if mpc is not None:
                        mpc.brake()

                case _ParentMsgType.STATE, (timestamp_s, state):
                    # 收到车辆状态，执行 MPC 更新
                    # 检查是否有新消息（丢弃过时数据）或 MPC 未初始化

                    if pipe.poll() or not mpc:
                        continue

                    result = mpc.update(state, delta_time_s)
                    pipe.send((timestamp_s, state, result))
    except (EOFError, KeyboardInterrupt, OSError):
        pass
    except Exception as e:
        print(f"[LocalPlanner Worker] Error: {e}")
        
class LocalPlannerAdapter:
    """
    LocalPlanner Adapter
    
    This adapter:
    1. Uses ProcessAdapter to run MPC in a separate process
    2. Uses threading.Timer to periodically send vehicle state
    3. Subscribes to MEASURED_STATE events to get vehicle state
    4. Publishes events via EventBus instead of Qt Signals
    
    Events published:
    - LOCAL_PLANNER_CONTROL_SEQUENCE: (control_sequence: np.ndarray) - Control commands
    - LOCAL_PLANNER_TRAJECTORIES: (trajectories: LocalPlanningTrajectories) - Local planning trajectories
    """

    def __init__(
        self,
        event_bus: EventBus,
        delta_time_s: float,
        update_interval_s: float
    ) -> None:
        """
        Initialize local planner adapter
        
        Args:
            event_bus: EventBus instance for publishing events
            delta_time_s: Time step for MPC (seconds)
            update_interval_s: Interval for sending state updates (seconds)
        """
        self._event_bus = event_bus
        self._delta_time_s = delta_time_s
        self._update_interval = update_interval_s
        self._state: Optional[tuple[float, Car]] = None

        self._process_adapter = ProcessAdapter(
            event_bus=event_bus,
            event_name='_local_planner_worker_message',
            target=_worker_process,
            args=(delta_time_s,)
        )

        self._event_bus.subscribe('_local_planner_worker_message',  self._handle_worker_message)
        self._event_bus.subscribe(MEASURED_STATE, self._on_measured_state)

        self._update_timer: Optional[threading.Timer] = None
        self._running = False
        self._timer_lock = threading.RLock()

    def start(self) -> None:
        """Start the worker process and update timer"""
        if self._running:
            return 
        self._running = True

        self._process_adapter.start()
        self._schedule_update()

    def _on_measured_state(self, timestamp_s: float, state: Car) -> None:
        """
        Handle MEASURED_STATE events
        Store the latest vehicle state for periodic updates
        
        Args:
            timestamp_s: Simulation timestamp
            state: Current vehicle state
        """
        self._state = (timestamp_s, state)
        
        
    def _schedule_update(self) -> None:
        """
        Schedule the next state update
        Use threading.Timer for periodic invocation
        """
        with self._timer_lock:
            if not self._running:
                return
            
            if self._state is not None:
                self._process_adapter.send((_ParentMsgType.STATE, self._state))
            
            # schedule the next update
            self._update_timer = threading.Timer(
                self._update_interval,
                self._schedule_update
            )

            self._update_timer.daemon = True
            self._update_timer.start()

    def _handle_worker_message(self, data: tuple[float, Car, MPCResult]) -> None:
        """
        Handle messages from worker process
        
        This method is called when ProcessAdapter receives a message from child process.
        It processes the MPC result and publishes appropriate events.
        
        Args:
            data: Message from worker process (timestamp, state, mpc_result)
        """
        timestamp_s, state, result = data

        # calc ctrl-sequence ts
        timestamps = np.arange(len(result.controls)) * self._delta_time_s + timestamp_s
        # calc velocity sequences with accumulated accelerations
        velocities = state.velocity + np.cumsum(result.controls[:, 0] * self._delta_time_s) 
        # calc sterring sequence
        steers = result.controls[:, 1]

        # build ctrl-dequence [ts, v, st]
        control_sequence = np.column_stack((timestamps, velocities, steers))

        # publish ctrl-sequence event
        self._event_bus.emit(LOCAL_PLANNER_CONTROL_SEQUENCE, control_sequence)

        # construct local planning trajectory
        local_trajectories = LocalPlanningTrajectories(
            local_trajectory=result.states[:, :2],      # 位置 (x, y)
            reference_points=result.ref_states,          # 参考点
            brake_trajectory=result.brake_trajectory    # 刹车轨迹
        )

        # publish local planning trajectory event
        self._event_bus.emit(LOCAL_PLANNER_TRAJECTORIES, local_trajectories)
        
    def set_trajectory(self, trajectory: Optional[npt.NDArray[np.floating[Any]]]) -> None:
        """
        Set the global trajectory for local planning
        
        Args:
            trajectory: Global trajectory (numpy array) or None to brake
        """
        if trajectory is not None:
            self._process_adapter.send((_ParentMsgType.TRAJECTORY, trajectory))
        else:
            self._process_adapter.send(_ParentMsgType.BRAKE)
    
    def brake(self) -> None:
        """Send brake command to local planner"""
        self._process_adapter.send(_ParentMsgType.BRAKE)
    
    def cancel(self) -> None:
        """Cancel local planning"""
        self._process_adapter.send(_ParentMsgType.CANCEL)
    
    def stop(self) -> None:
        """Stop the worker process, timer, and clean up resources"""
        self._running = False

        with self._timer_lock:
            if self._update_timer:
                self._update_timer.cancel()
                self._update_timer = None

        try:
            self._event_bus.unsubscribe('_local_planner_worker_message', self._handle_worker_message)
            self._event_bus.unsubscribe(MEASURED_STATE, self._on_measured_state)
        except Exception as e:
            print(f"[LocalPlannerAdapter] Error unsubscribing: {e}")

        self._process_adapter.stop()
    
    def is_alive(self) -> bool:
        """
        Check if worker process is running
        
        Returns:
            True if process is alive, False otherwise
        """
        return self._process_adapter.is_alive()

    