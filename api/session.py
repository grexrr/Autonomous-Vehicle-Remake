import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from AutonomousVehicle.modeling.car import Car
from AutonomousVehicle.modeling.obstacles import Obstacles
from api.adapters.global_planner_adapter import GlobalPlannerAdapter
from api.adapters.local_planner_adapter import LocalPlannerAdapter
from api.adapters.trajectory_collision_checking_adapter import TrajectoryCollisionCheckingAdapter
from api.utils import serialize_car, serialize_obstacles, serialize_trajectory
# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AutonomousVehicle.constants import *

from api.adapters.car_simulation import CarSimulationAdapter
from api.adapters.event_bus import EventBus
from api.adapters.map_server import MapServerAdapter
from api.event_types import (
    MEASURED_STATE, MAP_INITIALIZED, KNOWN_OBSTACLES_UPDATED, NEW_OBSTACLES_DISCOVERED,
    GLOBAL_PLANNER_DISPLAY_SEGMENTS, GLOBAL_PLANNER_TRAJECTORY, GLOBAL_PLANNER_FINISHED,
    LOCAL_PLANNER_CONTROL_SEQUENCE, LOCAL_PLANNER_TRAJECTORIES,
    TRAJECTORY_COLLIDED,
    WS_STATE_UPDATE, WS_GLOBAL_TRAJECTORY, WS_GOAL_UNREACHABLE,
    WS_LOCAL_TRAJECTORIES, WS_OBSTACLES_UPDATED, WS_NEW_OBSTACLES, WS_DISPLAY_SEGMENTS
)

REPLAN_MAX_SPEED = 5 / 3.6 

class UserSession:
    """
    User session that manages a complete simulation instance
    
    This class:
    1. Creates and manages all adapters (MapServer, CarSimulation, GlobalPlanner, LocalPlanner)
    2. Connects events via EventBus (replacing Qt Signal/Slot)
    3. Provides control interfaces (set_goal, set_state, brake, cancel, etc.)
    4. Handles trajectory collision checking
    
    Each UserSession represents one independent simulation instance for one user.
    """

    def __init__(self, session_id: str, initial_state: Optional[dict] = None, map_name: str = "map2") -> None:
        """
        Initialize user session
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self._socketio = None
        self.event_bus = EventBus()

        # map
        self._map_name = map_name

        # init adapters
        self.map_server = MapServerAdapter(self.event_bus)
        self.car_simulation = CarSimulationAdapter(
            event_bus=self.event_bus,
            delta_time_s=SIMULATION_DELTA_TIME,
            simulation_interval_s=SIMULATION_INTERVAL,
            publish_interval_s=SIMULATION_PUBLISH_INTERVAL
        )
        self.global_planner = GlobalPlannerAdapter(
            event_bus=self.event_bus,
            segment_collection_size=GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE
        )
        self.local_planner = LocalPlannerAdapter(
            event_bus=self.event_bus,
            delta_time_s=LOCAL_PLANNER_DELTA_TIME,
            update_interval_s=LOCAL_PLANNER_UPDATE_INTERVAL
        )
        self.collision_checker = TrajectoryCollisionCheckingAdapter(self.event_bus)

        # state management
        self._measured_state:Optional[Car] = None
        self._goal_state:Optional[Car] = None
        self._brake_trajectory: Optional[npt.NDArray[np.floating[Any]]] = None
        self._local_planning = False 
        self._is_initialized = False
        self._initial_state = initial_state

        self._setup_event_handlers()
        self._initialize()


    def _setup_event_handlers(self) -> None:
        """
        Connect all event handlers via EventBus
        This replaces the Qt Signal.connect() calls in MainWindow
        """
        # ========== Car Simulation ==========
    
        # Car simulation -> Local planner (vehicle state updates)
        # Note: Local planner already subscribes to MEASURED_STATE in its __init__
        
        # Car sim -> Map server (LIDAR scan)
        self.event_bus.subscribe(MEASURED_STATE, self.map_server.update_from_vehicle_state)
        
        # Car sim -> Store measured state for replanning
        self.event_bus.subscribe(MEASURED_STATE, self._on_measured_state)

        # ========== Global Planner ==========

        # Global planner finished -> Car simulation resume
        self.event_bus.subscribe(GLOBAL_PLANNER_FINISHED, self.car_simulation.resume)

        # Global planner trajectory -> Local planner
        self.event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, self.local_planner.set_trajectory)

        self.event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, self._on_global_planning_result)

        # ========== Global Planner ==========
        # Local planner -> Car simulation (control sequence)
        self.event_bus.subscribe(
            LOCAL_PLANNER_CONTROL_SEQUENCE,
            self.car_simulation.set_control_sequence
        )
        # Local planner -> Store brake trajectory
        self.event_bus.subscribe(LOCAL_PLANNER_TRAJECTORIES, self._on_local_planning_trajectories)

        # ========== Map Server ==========
        # Map init -> Mark as initialized
        self.event_bus.subscribe(MAP_INITIALIZED, self._on_map_initialized)

        # ========== Collision Checker ==========
        # Collision detected -> Local planner brake
        self.event_bus.subscribe(TRAJECTORY_COLLIDED, self.local_planner.brake)
    
        # Collision detected -> Replan
        self.event_bus.subscribe(TRAJECTORY_COLLIDED, self._on_trajectory_collided)

    def _initialize(self) -> None:
        """
        Initialize map and start all components
        """
        # Initialize map (this will trigger MAP_INITIALIZED event)
        self.map_server.init_map(self._map_name)
        # Start all components
        self.car_simulation.start()
        self.global_planner.start()
        self.local_planner.start()
    
    def _on_measured_state(self, timestamp_s: float, state:Car) -> None:
        """
        Store the latest measured vehicle state
        
        Args:
            timestamp_s: Simulation timestamp
            state: Current vehicle state
        """
        self._measured_state = state

    def _on_trajectory_collided(self) -> None:
        """Handle trajectory collision - replan"""
        self.local_planner.brake()
        
        start = self._measured_state
        if start is None:
            return
        
        if self._goal_state is None:
            return
        
        if abs(start.velocity) > REPLAN_MAX_SPEED and self._brake_trajectory is not None:
            start = self._brake_trajectory
        
        coords = self.map_server.known_obstacle_coordinates
        if coords is None:
            return
        
        obstacles = Obstacles(coords)
        self.global_planner.plan(start, self._goal_state, obstacles)
        
    def _on_global_planning_result(self, trajectory: Optional[npt.NDArray[np.floating[Any]]]) -> None:
        """
        Handle global planning result
        
        Args:
            trajectory: Planned trajectory or None if unreachable
        """
        if trajectory is not None:
            self._local_planning = True
        else:
            self._local_planning = False

    def _on_local_planning_trajectories(self, trajectories) -> None:
        """
        Store brake trajectory from local planner
        
        Args:
            trajectories: LocalPlanningTrajectories named tuple
        """
        if self._local_planning:
            self._brake_trajectory = trajectories.brake_trajectory

    def _on_map_initialized(self) -> None:
        """Handle map initialization complete"""
        self._is_initialized = True
        if self._initial_state is not None:
            try:
                self.set_state(
                    self._initial_state['x'],
                    self._initial_state['y'],
                    self._initial_state['yaw']
                )
            except (KeyError, TypeError) as e:
                print(f"Warning: Invalid initial_state format, using random state: {e}")
                random_state = self.map_server.generate_random_initial_state()
                self.set_state(random_state.x, random_state.y, random_state.yaw)
        else:
            random_state = self.map_server.generate_random_initial_state()
            self.set_state(random_state.x, random_state.y, random_state.yaw)

    def set_state(self, x: float, y: float, yaw: float) -> None:
        """
        Set vehicle initial state
        
        Args:
            x: Initial x coordinate (meters)
            y: Initial y coordinate (meters)
            yaw: Initial yaw angle (radians)
        """
        # Clear destination and cancel ongoing planning
        self._goal_state = None
        self._local_planning = False
        self._brake_trajectory = None
        self.global_planner.cancel()
        self.local_planner.brake()
        self.collision_checker.cancel()

        state = Car(x, y, yaw)
        self.car_simulation.set_state(state)

        # reset scan state and scan around the new start position
        self.map_server.reset_discovery(state)
        
    
    def set_goal(self, x: float, y: float, yaw: float) -> None:
        """
        Set goal position and trigger global planning
        
        Args:
            x: Goal x coordinate (meters)
            y: Goal y coordinate (meters)
            yaw: Goal yaw angle (radians)
        """
        if not self._is_initialized:
            raise RuntimeError("Map not initialized yet")
        
        self.car_simulation.resume()
        self._goal_state = Car(x, y, yaw)
        
        start = self._measured_state
        if start is None:
            return
        
        if abs(start.velocity) > REPLAN_MAX_SPEED and self._brake_trajectory is not None:
            start = self._brake_trajectory
        
        coords = self.map_server.known_obstacle_coordinates
        if coords is None:
            return
        
        obstacles = Obstacles(coords)
        self.global_planner.plan(start, self._goal_state, obstacles)

    def brake(self) -> None:
        """
        Apply brake command
        """
        self._local_planning = False
        self.global_planner.cancel()
        self.local_planner.brake()
        self.collision_checker.cancel()
      
    def resume(self) -> None:
        """
        Resume simulation after brake
        If there's a goal state, replan from current state
        """
        if not self._is_initialized:
            raise RuntimeError("Map not initialized yet")
        
        # 恢复车辆仿真
        self.car_simulation.resume()
        
        # 如果有目标状态，重新规划
        if self._goal_state is not None:
            start = self._measured_state
            if start is None:
                return
            
            if abs(start.velocity) > REPLAN_MAX_SPEED and self._brake_trajectory is not None:
                start = self._brake_trajectory
            
            coords = self.map_server.known_obstacle_coordinates
            if coords is None:
                return
            
            obstacles = Obstacles(coords)
            self.global_planner.plan(start, self._goal_state, obstacles)

    def cancel(self) -> None:
        """
        Cancel all planning and stop simulation
        """
        self._local_planning = False
        self._brake_trajectory = None
        self.car_simulation.stop()
        self.global_planner.cancel()
        self.local_planner.cancel()
        self.collision_checker.cancel()

    def restart(self) -> None:
        """
        Restart simulation (reinitialize map)
        """
        self.cancel()
        self._brake_trajectory = None
        self._goal_state = None
        self.map_server.init_map(self._map_name)
        self.car_simulation.start()
        self.global_planner.start()
        self.local_planner.start()

    def get_state(self) -> Optional[tuple[float, Car]]:
        """
        Get current vehicle state
        
        Returns:
            (timestamp, car_state) or None if not available
        """
        return self.car_simulation.get_state()

    def get_map_data(self) -> dict:
        """
        Get map data for visualization
        
        Returns:
            Dictionary containing map information and vehicle parameters
        """
        known_coords = self.map_server.known_obstacle_coordinates
        unknown_coords = self.map_server.unknown_obstacle_coordinates
        return {
            'bounding_box': self.map_server.bounding_box,
            'known_obstacles': serialize_obstacles(known_coords),
            'unknown_obstacles': serialize_obstacles(unknown_coords),
            'vehicle_params': {
                'length': float(Car.LENGTH),
                'width': float(Car.WIDTH),
                'wheel_base': float(Car.WHEEL_BASE),
                'wheel_length': float(Car.WHEEL_LENGTH),
                'wheel_width': float(Car.WHEEL_WIDTH),
                'wheel_spacing': float(Car.WHEEL_SPACING),
                'back_to_wheel': float(Car.BACK_TO_WHEEL),
                'back_to_center': float(Car.BACK_TO_CENTER),
                'scan_radius': float(Car.SCAN_RADIUS),
                'collision_length': float(Car.COLLISION_LENGTH),
                'collision_width': float(Car.COLLISION_WIDTH),
                'collision_radius': float(Car.COLLISION_RADIUS)
            }
        }

    def stop(self) -> None:
        """
        Stop all components and clean up resources
        """
        # Stop all adapters
        self.car_simulation.stop()
        self.global_planner.stop()
        self.local_planner.stop()
        self.collision_checker.stop()
        
        # Clear state
        self._measured_state = None
        self._goal_state = None
        self._brake_trajectory = None
        self._is_initialized = False

    # =================== WebSocket Support =================== 

    def register_websocket_push(self, socketio_instance) -> None:
        """
        Register WebSocket push

        When the client connects, subscribe to events and push updates to the WebSocket.

        Args:
            socketio_instance: SocketIO instance
        """
        # 如果已经有 socketio 注册，说明是重连，不需要重新订阅事件
        is_reconnect = self._socketio is not None
        
        self._socketio = socketio_instance

        # 只有在首次连接时才订阅事件（避免重复订阅）
        if not is_reconnect:
            # subscribe car stat event, 推送车辆状态更新到WebSocket
            self.event_bus.subscribe(MEASURED_STATE, self._push_state_update)

            # global planning result
            self.event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, self._push_global_trajectory)

            # local planning trajectories
            self.event_bus.subscribe(LOCAL_PLANNER_TRAJECTORIES, self._push_local_trajectories)

            # update map obstacles update
            self.event_bus.subscribe(KNOWN_OBSTACLES_UPDATED, self._push_obstacles_updated)
            self.event_bus.subscribe(NEW_OBSTACLES_DISCOVERED, self._push_new_obstacles)

            # display planning
            self.event_bus.subscribe(GLOBAL_PLANNER_DISPLAY_SEGMENTS, self._push_display_segments)
    
    def unregister_websocket_push(self) -> None:
        """
        Unregister WebSocket push
        
        Called when client disconnects to clean up the connection reference.
        Note: Event bus subscriptions are not unsubscribed here as they may be needed
        for other purposes. The _socketio reference is cleared to allow reconnection detection.
        """
        self._socketio = None
    
    def _push_state_update(self, timestamp_s: float, state: Car) -> None:
        """Push vehicle state update to WebSocket"""
        if self._socketio is None:
            return
        
        self._socketio.emit(WS_STATE_UPDATE, {
            'timestamp': timestamp_s,
            'car': serialize_car(state)
        }, room=self.session_id)
    
    def _push_global_trajectory(self, trajectory) -> None:
        """Push global planner trajectory to WebSocket"""
        if self._socketio is None:
            return
        
        if trajectory is not None:
            self._socketio.emit(WS_GLOBAL_TRAJECTORY, {
                'trajectory': serialize_trajectory(trajectory)
            }, room=self.session_id)
        else:
            self._socketio.emit(WS_GOAL_UNREACHABLE, {
                'message': 'Goal is unreachable'
            }, room=self.session_id)
    
    def _push_local_trajectories(self, trajectories) -> None:
        """Push local planner trajectories to WebSocket"""
        if self._socketio is None:
            return
    
        self._socketio.emit(WS_LOCAL_TRAJECTORIES, {
            'local_trajectory': serialize_trajectory(trajectories.local_trajectory),
            'reference_points': serialize_trajectory(trajectories.reference_points),
            'brake_trajectory': serialize_trajectory(trajectories.brake_trajectory)
        }, room=self.session_id)
        
    def _push_obstacles_updated(self, obstacles) -> None:
        """Push updated obstacles to WebSocket"""
        if self._socketio is None:
            return
        
        self._socketio.emit(WS_OBSTACLES_UPDATED, {
            'obstacles': serialize_obstacles(obstacles)
        }, room=self.session_id)
    
    def _push_new_obstacles(self, new_obstacles) -> None:
        """Push newly found obstacles to WebSocket"""
        if self._socketio is None:
            return
        
        from api.utils import serialize_obstacles
        
        self._socketio.emit(WS_NEW_OBSTACLES, {
            'obstacles': serialize_obstacles(new_obstacles)
        }, room=self.session_id)
    
    def _push_display_segments(self, display_segments) -> None:
        """Push intermediate global planning segments to WebSocket"""
        if self._socketio is None:
            return
        
        serialized_segments = [serialize_trajectory(seg) for seg in display_segments]

        self._socketio.emit(WS_DISPLAY_SEGMENTS, {
            'segments': serialized_segments
        }, room=self.session_id)
