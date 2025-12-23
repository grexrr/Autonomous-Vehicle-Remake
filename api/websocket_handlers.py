from multiprocessing import managers
from flask import request
from flask_socketio import disconnect, emit, join_room

from api import session
from api.simulation_manager import SimulationManager
from api.utils import serialize_car

socketio = None

def init_websocket_handlers(socketio_instance):
    """
    Initialize WebSocket handlers

    This function is called in app.py, passing the socketio instance.
    Since socketio is created in app.py, delayed import is required.
    """
    global socketio
    socketio = socketio_instance
    register_handlers()

def register_handlers():
    """Register all WebSocket event handlers"""
    if socketio is None:
        raise RuntimeError("SocketIO instance not initialized.")
    
    @socketio.on('connect')
    def handle_connect(auth):
        """Handle Client Connection"""
        
        session_id = None

        if isinstance(auth, dict):
            session_id = auth.get('session_id')
        elif isinstance(auth, str):
            session_id = auth
        
        if not session_id:
            session_id = request.args.get('session_id')

        import json, time
        with open('/Users/grexrr/Documents/Autonomous-Vehicle-Remake/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'id': 'log_ws_connect',
                'timestamp': time.time() * 1000,
                'location': 'websocket_handlers.py:connect',
                'data': {'session_id': session_id, 'auth': auth},
                'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'E'
            }) + '\n')
        
        if not session_id:
            emit('error', {'message': 'session_id is requrired'})
            disconnect()
            return False
    
        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit('error', {'message': f'Session {session_id} not found'})
            disconnect()
            return False
        
        join_room(session_id)

        session.register_websocket_push(socketio)
        emit('connected', {
            'session_id': session_id,
            'message': 'Connected successfully'
        })

        state = session.get_state()
        if state:
            ts, c = state
            emit('state_update', {
                'timestamp': ts,
                'car': serialize_car(c)
            })
        
        map_data = session.get_map_data()
        emit('map_data', map_data)

        return True
    
    @socketio.on('disconnect')
    def handle_disconnect(reason=None):
        import json, time
        with open('/Users/grexrr/Documents/Autonomous-Vehicle-Remake/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'id': 'log_ws_disconnect',
                'timestamp': time.time() * 1000,
                'location': 'websocket_handlers.py:disconnect',
                'message': 'client disconnected',
                'data': {'reason': reason, 'session_id': request.args.get('session_id')},
                'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'E'
            }) + '\n')

    @socketio.on('set_goal')
    def handle_set_goal(data):
        """Handle set destination goal"""
        session_id = data.get('session_id')
        if not session_id:
            emit('error', {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit('error', {'message': f'Session {session_id} not found'})
            return
        
        try:
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            yaw = float(data.get('yaw', 0.0))

            session.set_goal(x, y, yaw)

            emit ('goal_set', {
                'x': x,
                'y': y,
                'yaw': yaw,
                'message': 'Goal set successfully'
            })
        except Exception as e:
            emit('error', {'message': f'Session {session_id} failed to set goal: {str(e)}'})

    @socketio.on('set_state')
    def handle_set_state(data):
        """Handle set vehicle state"""
        session_id = data.get('session_id')
        if not session_id:
            emit('error', {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit('error', {'message': f'Session {session_id} not found'})
            return
        
        try:
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            yaw = float(data.get('yaw', 0.0))
            
            session.set_state(x, y, yaw)
            
            emit('state_set', {
                'x': x,
                'y': y,
                'yaw': yaw,
                'message': 'State set successfully'
            })
        except Exception as e:
            emit('error', {'message': f'Session {session_id} failed to set state: {str(e)}'})
    
    @socketio.on('brake')
    def handle_brake(data=None):
        """Handle brake command"""
        session_id = None

        if data:
            session_id = data.get('session_id')
        else:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit('error', {'message': 'session_id is required'})
            return

        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit('error', {'message': f'Session {session_id} not found'})
            return
        
        try:
            session.brake()
            emit('braked', {'message': 'Brake applied'})
        except Exception as e:
            emit('error', {'message': f'Session {session_id} failed to brake: {str(e)}'})

    @socketio.on('cancel')
    def handle_cancel(data=None):
        """处理取消命令"""
        session_id = None
        
        if data:
            session_id = data.get('session_id')
        else:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit('error', {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)
        
        if session is None:
            emit('error', {'message': f'Session {session_id} not found'})
            return
        
        try:
            session.cancel()
            emit('canceled', {'message': 'Simulation canceled'})
        except Exception as e:
            emit('error', {'message': f'Session {session_id} failed to cancel: {str(e)}'})

    @socketio.on('restart')
    def handle_restart(data=None):
        """处理重启命令"""
        session_id = None
        
        if data:
            session_id = data.get('session_id')
        else:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit('error', {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)
        
        if session is None:
            emit('error', {'message': f'Session {session_id} not found'})
            return
        
        try:
            session.restart()
            emit('restarted', {'message': 'Simulation restarted'})
        except Exception as e:
            emit('error', {'message': f'Session {session_id} failed to restart: {str(e)}'})