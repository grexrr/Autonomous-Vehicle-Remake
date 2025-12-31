MEASURED_STATE = 'measured_state'
MAP_INITIALIZED = 'map_initialized'
KNOWN_OBSTACLES_UPDATED = 'known_obstacles_updated'
NEW_OBSTACLES_DISCOVERED = 'new_obstacles_discovered'

# global planner events
GLOBAL_PLANNER_DISPLAY_SEGMENTS = 'global_planner_display_segments'
GLOBAL_PLANNER_TRAJECTORY = 'global_planner_trajectory'
GLOBAL_PLANNER_FINISHED = 'global_planner_finished'

# local planner events
LOCAL_PLANNER_CONTROL_SEQUENCE = 'local_planner_control_sequence'
LOCAL_PLANNER_TRAJECTORIES = 'local_planner_trajectories'

TRAJECTORY_COLLIDED = 'trajectory_collided'

# WebSocket client-to-server events (events sent by client)
WS_CONNECT = 'connect'
WS_DISCONNECT = 'disconnect'
WS_SET_GOAL = 'set_goal'
WS_SET_STATE = 'set_state'
WS_BRAKE = 'brake'
WS_CANCEL = 'cancel'
WS_RESTART = 'restart'
WS_RESUME = 'resume'
WS_CLOSE_SESSION = 'close_session'

# WebSocket server-to-client events (events emitted by server)
WS_ERROR = 'error'
WS_CONNECTED = 'connected'
WS_RECONNECTED = 'reconnected'
WS_RECONNECT_ATTEMPT = 'reconnect_attempt'
WS_STATE_UPDATE = 'state_update'
WS_MAP_DATA = 'map_data'
WS_GOAL_SET = 'goal_set'
WS_STATE_SET = 'state_set'
WS_BRAKED = 'braked'
WS_CANCELED = 'canceled'
WS_RESTARTED = 'restarted'
WS_RESUMED = 'resumed'
WS_SESSION_CLOSED = 'session_closed'
WS_GLOBAL_TRAJECTORY = 'global_trajectory'
WS_GOAL_UNREACHABLE = 'goal_unreachable'
WS_LOCAL_TRAJECTORIES = 'local_trajectories'
WS_OBSTACLES_UPDATED = 'obstacles_updated'
WS_NEW_OBSTACLES = 'new_obstacles'
WS_DISPLAY_SEGMENTS = 'display_segments'
