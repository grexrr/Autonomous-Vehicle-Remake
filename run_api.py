# run_api.py
"""本地开发启动脚本"""
from api.app import create_app
from dotenv import load_dotenv 
import os

load_dotenv()
if __name__ == '__main__':
    app = create_app('development')
    port = int(os.getenv('PORT', 5000))  
    
    print(f"🚀 服务器启动在 http://0.0.0.0:{port}")
    print(f"📡 健康检查: http://localhost:{port}/api/vehicle/health")  # 修正路径
    
    # 从 app 获取 socketio（因为我们在 app.py 中设置了 app.socketio = socketio）
    socketio = app.socketio
    socketio.run(app, host='0.0.0.0', port=port, debug=True)