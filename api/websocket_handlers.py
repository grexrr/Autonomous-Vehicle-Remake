import threading
from flask import request
from flask_socketio import disconnect, emit, join_room

from api.event_types import (
    WS_CONNECT, WS_DISCONNECT, WS_SET_GOAL, WS_SET_STATE, WS_BRAKE, WS_CANCEL, WS_RESTART, WS_RESUME,
    WS_ERROR, WS_CONNECTED, WS_RECONNECTED, WS_STATE_UPDATE, WS_MAP_DATA, WS_GOAL_SET, WS_STATE_SET,
    WS_BRAKED, WS_CANCELED, WS_RESTARTED, WS_RESUMED
)
from api.simulation_manager import SimulationManager
from api.utils import serialize_car

socketio = None

# 映射 WebSocket sid 到 session_id，用于断开连接时清理资源
_sid_to_session_id: dict[str, str] = {}
_sid_lock = threading.RLock()

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
    
    @socketio.on(WS_CONNECT)
    def handle_connect(auth):
        """Handle Client Connection"""
        
        session_id = None

        if isinstance(auth, dict):
            session_id = auth.get('session_id')
        elif isinstance(auth, str):
            session_id = auth
        
        if not session_id:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit(WS_ERROR, {'message': 'session_id is requrired'})
            disconnect()
            return False
    
        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit(WS_ERROR, {'message': f'Session {session_id} not found'})
            disconnect()
            return False
        
        join_room(session_id)

        # 建立 sid 到 session_id 的映射，用于断开连接时清理
        sid = getattr(request, 'sid', None)
        if sid:
            with _sid_lock:
                _sid_to_session_id[sid] = session_id

        # 检测是否是重连：如果 session 已经有 socketio 注册，说明是重连
        is_reconnect = session._socketio is not None
        
        session.register_websocket_push(socketio)
        
        if is_reconnect:
            # 重连成功，发送重连事件
            emit(WS_RECONNECTED, {
                'session_id': session_id,
                'message': 'Reconnected successfully'
            })
        else:
            # 首次连接
            emit(WS_CONNECTED, {
                'session_id': session_id,
                'message': 'Connected successfully'
            })

        # 重连时也需要重新发送当前状态和地图数据
        state = session.get_state()
        if state:
            ts, c = state
            emit(WS_STATE_UPDATE, {
                'timestamp': ts,
                'car': serialize_car(c)
            })
        
        map_data = session.get_map_data()
        emit(WS_MAP_DATA, map_data)

        return True
    
    @socketio.on(WS_DISCONNECT)
    def handle_disconnect(reason=None):
        """Handle client disconnection - clean up resources"""
        try:
            # 从映射中获取 session_id
            sid = getattr(request, 'sid', None)
            session_id = None
            if sid:
                with _sid_lock:
                    session_id = _sid_to_session_id.pop(sid, None)
            
            if session_id:
                manager = SimulationManager()
                session = manager.get_session(session_id)
                
                if session is not None:
                    # 清理 WebSocket 引用，允许下次正常连接
                    session.unregister_websocket_push()
        except Exception as e:
            # 断开时 request 可能不可用，需要异常处理
            # 使用 print 而不是 logging，因为这是关键错误
            print(f"Error in handle_disconnect: {e}")

    @socketio.on(WS_SET_GOAL)
    def handle_set_goal(data):
        """Handle set destination goal"""
        session_id = data.get('session_id')
        if not session_id:
            emit(WS_ERROR, {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit(WS_ERROR, {'message': f'Session {session_id} not found'})
            return
        
        try:
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            yaw = float(data.get('yaw', 0.0))

            session.set_goal(x, y, yaw)

            emit(WS_GOAL_SET, {
                'x': x,
                'y': y,
                'yaw': yaw,
                'message': 'Goal set successfully'
            })
        except Exception as e:
            emit(WS_ERROR, {'message': f'Session {session_id} failed to set goal: {str(e)}'})

    @socketio.on(WS_SET_STATE)
    def handle_set_state(data):
        """Handle set vehicle state"""
        session_id = data.get('session_id')
        if not session_id:
            emit(WS_ERROR, {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit(WS_ERROR, {'message': f'Session {session_id} not found'})
            return
        
        try:
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            yaw = float(data.get('yaw', 0.0))
            
            session.set_state(x, y, yaw)
            
            emit(WS_STATE_SET, {
                'x': x,
                'y': y,
                'yaw': yaw,
                'message': 'State set successfully'
            })
        except Exception as e:
            emit(WS_ERROR, {'message': f'Session {session_id} failed to set state: {str(e)}'})
    
    @socketio.on(WS_BRAKE)
    def handle_brake(data=None):
        """Handle brake command"""
        session_id = None

        if data:
            session_id = data.get('session_id')
        else:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit(WS_ERROR, {'message': 'session_id is required'})
            return

        manager = SimulationManager()
        session = manager.get_session(session_id)

        if session is None:
            emit(WS_ERROR, {'message': f'Session {session_id} not found'})
            return
        
        try:
            session.brake()
            emit(WS_BRAKED, {'message': 'Brake applied'})
        except Exception as e:
            emit(WS_ERROR, {'message': f'Session {session_id} failed to brake: {str(e)}'})

    @socketio.on(WS_CANCEL)
    def handle_cancel(data=None):
        """处理取消命令"""
        session_id = None
        
        if data:
            session_id = data.get('session_id')
        else:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit(WS_ERROR, {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)
        
        if session is None:
            emit(WS_ERROR, {'message': f'Session {session_id} not found'})
            return
        
        try:
            session.cancel()
            emit(WS_CANCELED, {'message': 'Simulation canceled'})
        except Exception as e:
            emit(WS_ERROR, {'message': f'Session {session_id} failed to cancel: {str(e)}'})

    @socketio.on(WS_RESTART)
    def handle_restart(data=None):
        """处理重启命令"""
        session_id = None
        
        if data:
            session_id = data.get('session_id')
        else:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit(WS_ERROR, {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)
        
        if session is None:
            emit(WS_ERROR, {'message': f'Session {session_id} not found'})
            return
        
        try:
            session.restart()
            emit(WS_RESTARTED, {'message': 'Simulation restarted'})
            map_data = session.get_map_data()
            emit(WS_MAP_DATA, map_data)
        except Exception as e:
            emit(WS_ERROR, {'message': f'Session {session_id} failed to restart: {str(e)}'})

    @socketio.on(WS_RESUME)
    def handle_resume(data=None):
        """处理继续命令"""
        session_id = None
        
        if data:
            session_id = data.get('session_id')
        else:
            session_id = request.args.get('session_id')
        
        if not session_id:
            emit(WS_ERROR, {'message': 'session_id is required'})
            return
        
        manager = SimulationManager()
        session = manager.get_session(session_id)
        
        if session is None:
            emit(WS_ERROR, {'message': f'Session {session_id} not found'})
            return
        
        try:
            session.resume()
            emit(WS_RESUMED, {'message': 'Simulation resumed'})
        except Exception as e:
            emit(WS_ERROR, {'message': f'Session {session_id} failed to resume: {str(e)}'})
