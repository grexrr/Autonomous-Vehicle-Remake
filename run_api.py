# run_api.py
from api.app import create_app
from dotenv import load_dotenv 
import os
import sys

load_dotenv()
if __name__ == '__main__':
    env = os.getenv('FLASK_ENV', 'development')

    app = create_app(env)
    port = int(os.getenv('PORT', 5000))  
    
    print("=" * 50)
    print("🚀 Autonomous Vehicle API Server")
    print("=" * 50)

    print(f"🚀 Server: http://0.0.0.0:{port}")
    print(f"🔍 Health Check: http://localhost:{port}/api/vehicle/health")
    print(f"🌐 WebSocket: ws://localhost:{port}")
    print(f"📝 Environment: {env}")
    print("=" * 50)
    print("Press CTRL+C to stop the server")
    print("=" * 50)
    
    try:
        socketio = app.socketio # type: ignore[attr-defined]
        debug_mode = (env == 'development')
        socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)