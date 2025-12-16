from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from .config import config

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
        # async_mode='eventlet' 
    )

    app.socketio = socketio

    # 4. register route
    from .routes import api_vehicle
    app.register_blueprint(api_vehicle)

    return app