# test_api.py（放在项目根目录）
"""测试 API 基础框架"""
from api.app import create_app

if __name__ == '__main__':
    app = create_app('development')
    print("Flask 应用创建成功！")
    print(f"配置的 CORS 域名: {app.config['CORS_ORIGINS']}")
    print(f"调试模式: {app.config['DEBUG']}")

    from api.app import socketio
    if socketio:
        print("SocketIO 初始化成功！")
    else:
        print("警告：SocketIO 未初始化")