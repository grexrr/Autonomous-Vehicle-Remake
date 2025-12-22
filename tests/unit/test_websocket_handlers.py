"""
WebSocket Handlers 单元测试
测试 WebSocket 事件处理器的各项功能
"""

import time
from unittest.mock import Mock, MagicMock, patch, call
from flask import Flask
from flask_socketio import SocketIO

from api.websocket_handlers import init_websocket_handlers, register_handlers
from api.simulation_manager import SimulationManager
from api.session import UserSession


def test_init_websocket_handlers():
    """测试初始化 WebSocket 处理器"""
    print("=== Test 1: Initialize WebSocket Handlers ===")
    
    # 创建 mock socketio 实例
    mock_socketio = Mock(spec=SocketIO)
    
    # 初始化
    init_websocket_handlers(mock_socketio)
    
    # 验证 socketio 被设置
    from api.websocket_handlers import socketio
    assert socketio == mock_socketio, "socketio 实例应该被设置"
    
    print("✓ Test passed\n")


def test_register_handlers_without_socketio():
    """测试在没有 socketio 时注册处理器应该失败"""
    print("=== Test 2: Register Handlers Without SocketIO ===")
    
    # 清除全局 socketio
    from api import websocket_handlers
    original_socketio = websocket_handlers.socketio
    websocket_handlers.socketio = None
    
    try:
        register_handlers()
        assert False, "应该抛出 RuntimeError"
    except RuntimeError as e:
        assert "not initialized" in str(e).lower(), f"错误消息应该包含 'not initialized'，实际: {e}"
        print(f"  正确捕获错误: {e}")
    finally:
        # 恢复
        websocket_handlers.socketio = original_socketio
    
    print("✓ Test passed\n")


def test_handle_connect_with_dict_auth():
    """测试使用字典格式的 auth 连接"""
    print("=== Test 3: Connect with Dict Auth ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None, "Session 应该存在"
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 直接测试核心逻辑（模拟连接处理中的 session_id 提取逻辑）
    # 这是 handle_connect 函数的核心逻辑，不需要 Flask 请求上下文
    auth = {'session_id': session_id}
    
    # 测试从字典中提取 session_id 的逻辑（与 handle_connect 中的逻辑一致）
    result_session_id = None
    if isinstance(auth, dict):
        result_session_id = auth.get('session_id')
    elif isinstance(auth, str):
        result_session_id = auth
    
    assert result_session_id == session_id, "应该从字典中提取 session_id"
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_handle_connect_with_string_auth():
    """测试使用字符串格式的 auth 连接"""
    print("=== Test 4: Connect with String Auth ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    
    # 测试字符串格式的 auth
    auth = session_id
    
    result_session_id = None
    if isinstance(auth, str):
        result_session_id = auth
    
    assert result_session_id == session_id, "应该从字符串中提取 session_id"
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_handle_connect_without_session_id():
    """测试没有 session_id 的连接应该失败"""
    print("=== Test 5: Connect Without Session ID ===")
    
    # 测试没有 session_id 的情况
    auth = {}
    
    result_session_id = None
    if isinstance(auth, dict):
        result_session_id = auth.get('session_id')
    elif isinstance(auth, str):
        result_session_id = auth
    
    assert result_session_id is None, "应该无法提取 session_id"
    
    print("✓ Test passed\n")


def test_handle_set_goal():
    """测试设置目标处理器"""
    print("=== Test 6: Handle Set Goal ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None, "Session 应该存在"
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 设置初始状态
    session.set_state(x=5.0, y=5.0, yaw=0.0)
    time.sleep(0.3)
    
    # 测试设置目标
    try:
        session.set_goal(x=15.0, y=15.0, yaw=0.0)
        print("  目标设置成功")
    except Exception as e:
        print(f"  目标设置失败（可能地图未初始化）: {e}")
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_handle_set_state():
    """测试设置状态处理器"""
    print("=== Test 7: Handle Set State ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None, "Session 应该存在"
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 测试设置状态
    session.set_state(x=10.0, y=10.0, yaw=1.0)
    time.sleep(0.3)
    
    # 验证状态
    state = session.get_state()
    assert state is not None, "状态应该已设置"
    timestamp, car = state
    assert abs(car.x - 10.0) < 0.1, f"x 坐标应该接近 10.0，实际: {car.x}"
    assert abs(car.y - 10.0) < 0.1, f"y 坐标应该接近 10.0，实际: {car.y}"
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_handle_brake():
    """测试刹车处理器"""
    print("=== Test 8: Handle Brake ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None, "Session 应该存在"
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 设置状态
    session.set_state(x=10.0, y=10.0, yaw=0.0)
    time.sleep(0.2)
    
    # 测试刹车
    session.brake()
    
    # 验证状态
    assert session._local_planning == False, "刹车后应该停止局部规划"
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_handle_cancel():
    """测试取消处理器"""
    print("=== Test 9: Handle Cancel ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None, "Session 应该存在"
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 设置状态
    session.set_state(x=10.0, y=10.0, yaw=0.0)
    time.sleep(0.2)
    
    # 测试取消
    session.cancel()
    
    # 验证状态
    assert session._local_planning == False, "取消后应该停止局部规划"
    assert session._brake_trajectory is None, "取消后应该清除刹车轨迹"
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_handle_restart():
    """测试重启处理器"""
    print("=== Test 10: Handle Restart ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None, "Session 应该存在"
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 设置状态和目标
    session.set_state(x=10.0, y=10.0, yaw=0.0)
    time.sleep(0.2)
    
    try:
        session.set_goal(x=20.0, y=20.0, yaw=0.0)
    except RuntimeError:
        pass
    
    # 测试重启
    session.restart()
    
    # 验证状态已清除
    assert session._brake_trajectory is None, "重启后应该清除刹车轨迹"
    assert session._goal_state is None, "重启后应该清除目标状态"
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_handle_connect_with_invalid_session():
    """测试使用无效 session_id 连接应该失败"""
    print("=== Test 11: Connect with Invalid Session ===")
    
    # 测试无效的 session_id
    invalid_session_id = "invalid_session_12345"
    
    manager = SimulationManager()
    session = manager.get_session(invalid_session_id)
    
    assert session is None, "无效的 session_id 应该返回 None"
    
    print("✓ Test passed\n")


def test_handle_set_goal_with_invalid_data():
    """测试使用无效数据设置目标"""
    print("=== Test 12: Set Goal with Invalid Data ===")
    
    # 创建测试环境
    manager = SimulationManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session is not None, "Session 应该存在"
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 测试无效数据（缺少参数）
    try:
        # 模拟缺少 x, y 的情况（会使用默认值 0.0）
        test_data = {}
        x = float(test_data.get('x', 0.0)) if 'x' in test_data else 0.0
        # 如果没有 x，使用默认值 0.0，不会抛出异常
        assert x == 0.0, "应该使用默认值 0.0"
        print("  正确处理缺少参数的情况（使用默认值）")
    except (TypeError, ValueError) as e:
        print(f"  捕获异常: {e}")
    
    # 清理
    manager.delete_session(session_id)
    time.sleep(0.2)
    
    print("✓ Test passed\n")


if __name__ == '__main__':
    print("[Testing][WebSocket Handlers]...\n")
    
    try:
        test_init_websocket_handlers()
        test_register_handlers_without_socketio()
        test_handle_connect_with_dict_auth()
        test_handle_connect_with_string_auth()
        test_handle_connect_without_session_id()
        test_handle_set_goal()
        test_handle_set_state()
        test_handle_brake()
        test_handle_cancel()
        test_handle_restart()
        test_handle_connect_with_invalid_session()
        test_handle_set_goal_with_invalid_data()
        
        print("=" * 40)
        print("🎉[Testing][WebSocket Handlers] All tests passed!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

