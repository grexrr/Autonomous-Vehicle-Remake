from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from .config import config
from .websocket_handlers import init_websocket_handlers
import os

# import eventlet
# print(f"[DEBUG] Eventlet version: {eventlet.__version__}")
# print(f"[DEBUG] Eventlet patched socket: {eventlet.patcher.is_monkey_patched('socket')}")


socketio = None

def create_app(config_name='development'):
    """
    Application factory function - creates and configures a Flask application.

    Args:
        config_name: Name of the configuration ('development' or 'production')

    Returns:
        Configured Flask application instance
    """
    global socketio
    # 1. init & load config
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # 2. config CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])

    # 3. init SocketIO (for WebSocket Transmission)
    socketio = SocketIO(
        app,
        cors_allowed_origins=app.config['SOCKETIO_CORS_ALLOWED_ORIGINS'],
        async_mode='threading',
        transports=['websocket'],  # force websocket to avoid long-poll fallback issues
        ping_timeout=300,          # allow longer idle before timing out
        ping_interval=20,           # send ping a bit more frequently
    )
    
    app.socketio = socketio # type: ignore[attr-defined]
    
    # 4. register route
    from .routes import api_vehicle
    app.register_blueprint(api_vehicle)

    # 5. register WebSocket Processor
    init_websocket_handlers(socketio)

    return app

import os
_app_env = os.getenv('FLASK_ENV', 'development')
app = create_app(_app_env)