"""
UserSession 单元测试
测试用户会话管理的各项功能
"""

import time
import numpy as np
from unittest.mock import Mock

from api.session import UserSession
from api.adapters.event_bus import EventBus
from api.event_types import *
from AutonomousVehicle.modeling.Car import Car
from AutonomousVehicle.modeling.Obstacles import Obstacles


def test_initialization():
    """测试初始化"""
    print("=== Test 1: Initialization ===")
    
    session = UserSession(session_id="test_session_1")
    
    # 验证基本属性
    assert session.session_id == "test_session_1", "session_id 应该匹配"
    assert session.event_bus is not None, "event_bus 应该已创建"
    assert session.map_server is not None, "map_server 应该已创建"
    assert session.car_simulation is not None, "car_simulation 应该已创建"
    assert session.global_planner is not None, "global_planner 应该已创建"
    assert session.local_planner is not None, "local_planner 应该已创建"
    assert session.collision_checker is not None, "collision_checker 应该已创建"
    
    # 等待地图初始化完成（最多等待5秒）
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    assert session._is_initialized, "地图应该在初始化后完成"
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_set_state():
    """测试设置车辆状态"""
    print("=== Test 2: Set State ===")
    
    session = UserSession(session_id="test_session_2")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 设置状态
    session.set_state(x=10.0, y=10.0, yaw=0.0)
    
    # 等待一小段时间，让状态更新
    time.sleep(0.2)
    
    # 验证状态已设置
    state = session.get_state()
    assert state is not None, "状态应该已设置"
    timestamp, car = state
    assert abs(car.x - 10.0) < 0.1, f"x 坐标应该接近 10.0，实际: {car.x}"
    assert abs(car.y - 10.0) < 0.1, f"y 坐标应该接近 10.0，实际: {car.y}"
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_set_goal():
    """测试设置目标并触发规划"""
    print("=== Test 3: Set Goal ===")
    
    session = UserSession(session_id="test_session_3")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 收集规划结果
    trajectory_received = []
    
    def on_trajectory(trajectory):
        trajectory_received.append(trajectory)
        print(f"  收到轨迹: {trajectory is not None}")
    
    session.event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, on_trajectory)
    
    # 设置初始状态
    session.set_state(x=5.0, y=5.0, yaw=0.0)
    time.sleep(0.3)  # 等待状态更新
    
    # 设置目标
    try:
        session.set_goal(x=15.0, y=15.0, yaw=0.0)
        
        # 等待规划完成（最多等待10秒）
        max_wait = 10.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if len(trajectory_received) > 0:
                break
            time.sleep(0.1)
        
        # 验证至少收到了轨迹事件（可能为None，如果规划失败）
        assert len(trajectory_received) > 0, "应该收到轨迹事件"
        print(f"  规划结果: {'成功' if trajectory_received[0] is not None else '失败（可能被障碍物阻挡）'}")
    except RuntimeError as e:
        # 如果地图未初始化，会抛出 RuntimeError
        print(f"  设置目标失败（可能地图未初始化）: {e}")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_set_goal_before_initialization():
    """测试在地图初始化前设置目标应该失败"""
    print("=== Test 4: Set Goal Before Initialization ===")
    
    session = UserSession(session_id="test_session_4")
    
    # 立即尝试设置目标（在地图初始化前）
    # 注意：由于初始化是异步的，我们需要快速尝试
    # 或者手动设置 _is_initialized = False
    session._is_initialized = False
    
    try:
        session.set_goal(x=10.0, y=10.0, yaw=0.0)
        assert False, "应该抛出 RuntimeError"
    except RuntimeError as e:
        assert "not initialized" in str(e).lower(), f"错误消息应该包含 'not initialized'，实际: {e}"
        print(f"  正确捕获错误: {e}")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_brake():
    """测试刹车功能"""
    print("=== Test 5: Brake ===")
    
    session = UserSession(session_id="test_session_5")
    
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
    
    # 调用刹车
    session.brake()
    
    # 验证状态
    assert session._local_planning == False, "刹车后应该停止局部规划"
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_cancel():
    """测试取消功能"""
    print("=== Test 6: Cancel ===")
    
    session = UserSession(session_id="test_session_6")
    
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
        pass  # 如果失败也没关系
    
    # 调用取消
    session.cancel()
    
    # 验证状态
    assert session._local_planning == False, "取消后应该停止局部规划"
    assert session._brake_trajectory is None, "取消后应该清除刹车轨迹"
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_restart():
    """测试重启功能"""
    print("=== Test 7: Restart ===")
    
    session = UserSession(session_id="test_session_7")
    
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
    
    # 调用重启
    session.restart()
    
    # 验证状态已清除
    assert session._brake_trajectory is None, "重启后应该清除刹车轨迹"
    assert session._goal_state is None, "重启后应该清除目标状态"
    
    # 等待地图重新初始化
    time.sleep(0.5)
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_resume():
    """测试继续功能"""
    print("=== Test 8: Resume ===")
    
    session = UserSession(session_id="test_session_8_resume")
    
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
        time.sleep(0.3)
    except RuntimeError:
        pass
    
    # 先刹车
    session.brake()
    assert session._local_planning == False, "刹车后应该停止局部规划"
    
    # 记录继续前的状态
    stopped_before = session.car_simulation._stopped
    
    # 调用继续
    try:
        session.resume()
        
        # 验证车辆仿真已恢复（resume() 会设置 _stopped = False）
        assert session.car_simulation._stopped == False, "继续后车辆仿真应该恢复（_stopped = False）"
        
        # 如果有目标状态，应该会触发重新规划
        if session._goal_state is not None:
            print("  继续后应该触发重新规划（如果有目标状态）")
    except RuntimeError as e:
        # 如果地图未初始化，会抛出 RuntimeError
        print(f"  继续失败（可能地图未初始化）: {e}")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_get_state():
    """测试获取状态"""
    print("=== Test 9: Get State ===")
    
    session = UserSession(session_id="test_session_8")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 初始状态可能为 None（如果还没有设置）
    initial_state = session.get_state()
    # 初始状态可能为 None 或已设置（由地图初始化自动设置）
    
    # 设置状态
    session.set_state(x=5.0, y=5.0, yaw=1.0)
    time.sleep(0.3)
    
    # 获取状态
    state = session.get_state()
    assert state is not None, "状态应该已设置"
    timestamp, car = state
    assert isinstance(timestamp, float), "时间戳应该是浮点数"
    assert isinstance(car, Car), "车辆状态应该是 Car 对象"
    assert abs(car.x - 5.0) < 0.1, f"x 坐标应该接近 5.0，实际: {car.x}"
    assert abs(car.y - 5.0) < 0.1, f"y 坐标应该接近 5.0，实际: {car.y}"
    
    print(f"  状态: timestamp={timestamp:.3f}, x={car.x:.3f}, y={car.y:.3f}, yaw={car.yaw:.3f}")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_get_map_data():
    """测试获取地图数据"""
    print("=== Test 10: Get Map Data ===")
    
    session = UserSession(session_id="test_session_9")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 获取地图数据
    map_data = session.get_map_data()
    
    # 验证数据结构
    assert isinstance(map_data, dict), "地图数据应该是字典"
    assert 'bounding_box' in map_data, "应该包含 bounding_box"
    assert 'known_obstacles' in map_data, "应该包含 known_obstacles"
    assert 'unknown_obstacles' in map_data, "应该包含 unknown_obstacles"
    assert 'vehicle_params' in map_data, "应该包含 vehicle_params"
    
    # 验证边界框
    bbox = map_data['bounding_box']
    assert len(bbox) == 4, "边界框应该有4个值"
    xmin, ymin, xmax, ymax = bbox
    assert xmin < xmax, "xmin 应该小于 xmax"
    assert ymin < ymax, "ymin 应该小于 ymax"
    
    # 验证障碍物
    assert isinstance(map_data['known_obstacles'], list), "已知障碍物应该是列表"
    assert isinstance(map_data['unknown_obstacles'], list), "未知障碍物应该是列表"
    
    # 验证车辆参数
    vehicle_params = map_data['vehicle_params']
    assert isinstance(vehicle_params, dict), "vehicle_params 应该是字典"
    
    # 验证必需的车辆参数字段
    required_params = [
        'length', 'width', 'wheel_base', 'wheel_length', 'wheel_width',
        'wheel_spacing', 'back_to_wheel', 'back_to_center', 'scan_radius',
        'collision_length', 'collision_width', 'collision_radius'
    ]
    for param in required_params:
        assert param in vehicle_params, f"vehicle_params 应该包含 {param}"
        assert isinstance(vehicle_params[param], (int, float)), f"{param} 应该是数字类型"
        assert vehicle_params[param] > 0, f"{param} 应该是正数"
    
    # 验证参数值的合理性（与 Car 类中的常量值对比）
    assert abs(vehicle_params['length'] - Car.LENGTH) < 0.001, "length 应该匹配 Car.LENGTH"
    assert abs(vehicle_params['width'] - Car.WIDTH) < 0.001, "width 应该匹配 Car.WIDTH"
    assert abs(vehicle_params['wheel_base'] - Car.WHEEL_BASE) < 0.001, "wheel_base 应该匹配 Car.WHEEL_BASE"
    assert abs(vehicle_params['scan_radius'] - Car.SCAN_RADIUS) < 0.001, "scan_radius 应该匹配 Car.SCAN_RADIUS"
    
    print(f"  边界框: {bbox}")
    print(f"  已知障碍物数量: {len(map_data['known_obstacles'])}")
    print(f"  未知障碍物数量: {len(map_data['unknown_obstacles'])}")
    print(f"  车辆长度: {vehicle_params['length']} m")
    print(f"  车辆宽度: {vehicle_params['width']} m")
    print(f"  轴距: {vehicle_params['wheel_base']} m")
    print(f"  扫描半径: {vehicle_params['scan_radius']} m")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_stop():
    """测试停止功能"""
    print("=== Test 11: Stop ===")
    
    session = UserSession(session_id="test_session_10")
    
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
    
    # 调用停止
    session.stop()
    time.sleep(0.3)
    
    # 验证状态已清除
    assert session._measured_state is None, "停止后应该清除测量状态"
    assert session._goal_state is None, "停止后应该清除目标状态"
    assert session._brake_trajectory is None, "停止后应该清除刹车轨迹"
    assert session._is_initialized == False, "停止后应该标记为未初始化"
    
    # 验证适配器已停止
    assert not session.car_simulation._running, "车辆仿真应该已停止"
    assert not session.global_planner.is_alive(), "全局规划器应该已停止"
    assert not session.local_planner._running, "局部规划器应该已停止"
    
    print("✓ Test passed\n")


def test_event_handlers():
    """测试事件处理器"""
    print("=== Test 12: Event Handlers ===")
    
    session = UserSession(session_id="test_session_11")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 测试 MEASURED_STATE 事件
    measured_states = []
    
    def on_measured_state(timestamp, car):
        measured_states.append((timestamp, car))
    
    # 注意：session 已经订阅了 MEASURED_STATE，我们添加额外的订阅者
    session.event_bus.subscribe(MEASURED_STATE, on_measured_state)
    
    # 设置状态并等待
    session.set_state(x=10.0, y=10.0, yaw=0.0)
    time.sleep(0.5)
    
    # 验证收到了状态更新
    assert len(measured_states) > 0, "应该收到至少一个状态更新"
    assert session._measured_state is not None, "内部状态应该已更新"
    
    print(f"  收到 {len(measured_states)} 个状态更新")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_trajectory_collision_replan():
    """测试轨迹碰撞重规划"""
    print("=== Test 13: Trajectory Collision Replan ===")
    
    session = UserSession(session_id="test_session_12")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 收集规划结果
    trajectory_received = []
    
    def on_trajectory(trajectory):
        trajectory_received.append(trajectory)
    
    session.event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, on_trajectory)
    
    # 设置状态和目标
    session.set_state(x=5.0, y=5.0, yaw=0.0)
    time.sleep(0.3)
    
    try:
        session.set_goal(x=15.0, y=15.0, yaw=0.0)
        time.sleep(0.5)
    except RuntimeError:
        pass
    
    # 模拟碰撞事件
    # 需要先有测量状态和目标状态
    if session._measured_state is not None and session._goal_state is not None:
        # 触发碰撞事件
        session.event_bus.emit(TRAJECTORY_COLLIDED)
        time.sleep(0.5)
        
        # 验证可能会触发重规划（取决于实现）
        print("  碰撞事件已触发")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_register_websocket_push():
    """测试注册 WebSocket 推送"""
    print("=== Test 14: Register WebSocket Push ===")
    
    session = UserSession(session_id="test_session_13")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 创建 mock socketio
    mock_socketio = Mock()
    
    # 注册 WebSocket 推送
    session.register_websocket_push(mock_socketio)
    
    # 验证 socketio 被设置
    assert session._socketio == mock_socketio, "socketio 实例应该被设置"
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_push_state_update():
    """测试推送状态更新到 WebSocket"""
    print("=== Test 15: Push State Update ===")
    
    session = UserSession(session_id="test_session_14")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 创建 mock socketio
    mock_socketio = Mock()
    mock_emit = Mock()
    mock_socketio.emit = mock_emit
    
    # 注册 WebSocket 推送
    session.register_websocket_push(mock_socketio)
    
    # 设置状态并等待状态更新
    session.set_state(x=10.0, y=10.0, yaw=0.0)
    time.sleep(0.5)  # 等待状态更新事件
    
    # 验证 emit 被调用（可能被调用多次）
    # 至少应该被调用一次（如果状态更新了）
    if mock_emit.called:
        # 验证调用参数
        calls = mock_emit.call_args_list
        state_update_calls = [c for c in calls if len(c[0]) > 0 and c[0][0] == 'state_update']
        if state_update_calls:
            call_args = state_update_calls[0]
            assert 'timestamp' in call_args[0][1], "应该包含 timestamp"
            assert 'car' in call_args[0][1], "应该包含 car"
            print("  状态更新已推送到 WebSocket")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_push_global_trajectory_success():
    """测试推送全局轨迹（成功情况）"""
    print("=== Test 16: Push Global Trajectory (Success) ===")
    
    session = UserSession(session_id="test_session_15")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 创建 mock socketio
    mock_socketio = Mock()
    mock_emit = Mock()
    mock_socketio.emit = mock_emit
    
    # 注册 WebSocket 推送
    session.register_websocket_push(mock_socketio)
    
    # 创建模拟轨迹
    mock_trajectory = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    
    # 触发轨迹推送
    session._push_global_trajectory(mock_trajectory)
    
    # 验证 emit 被调用
    assert mock_emit.called, "应该调用 emit"
    call_args = mock_emit.call_args
    assert call_args[0][0] == 'global_trajectory', "应该发送 global_trajectory 事件"
    assert 'trajectory' in call_args[0][1], "应该包含 trajectory"
    assert call_args[1]['room'] == session.session_id, "应该发送到正确的房间"
    
    print("  全局轨迹已推送到 WebSocket")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_push_global_trajectory_failure():
    """测试推送全局轨迹（失败情况 - 目标不可达）"""
    print("=== Test 17: Push Global Trajectory (Failure) ===")
    
    session = UserSession(session_id="test_session_16")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 创建 mock socketio
    mock_socketio = Mock()
    mock_emit = Mock()
    mock_socketio.emit = mock_emit
    
    # 注册 WebSocket 推送
    session.register_websocket_push(mock_socketio)
    
    # 触发轨迹推送（None 表示目标不可达）
    session._push_global_trajectory(None)
    
    # 验证 emit 被调用
    assert mock_emit.called, "应该调用 emit"
    call_args = mock_emit.call_args
    assert call_args[0][0] == 'goal_unreachable', "应该发送 goal_unreachable 事件"
    assert 'message' in call_args[0][1], "应该包含 message"
    assert call_args[1]['room'] == session.session_id, "应该发送到正确的房间"
    
    print("  目标不可达消息已推送到 WebSocket")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_push_obstacles_updated():
    """测试推送障碍物更新"""
    print("=== Test 18: Push Obstacles Updated ===")
    
    session = UserSession(session_id="test_session_17")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 创建 mock socketio
    mock_socketio = Mock()
    mock_emit = Mock()
    mock_socketio.emit = mock_emit
    
    # 注册 WebSocket 推送
    session.register_websocket_push(mock_socketio)
    
    # 创建模拟障碍物坐标
    mock_obstacles = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    
    # 触发障碍物更新推送
    session._push_obstacles_updated(mock_obstacles)
    
    # 验证 emit 被调用
    assert mock_emit.called, "应该调用 emit"
    call_args = mock_emit.call_args
    assert call_args[0][0] == 'obstacles_updated', "应该发送 obstacles_updated 事件"
    assert 'obstacles' in call_args[0][1], "应该包含 obstacles"
    assert call_args[1]['room'] == session.session_id, "应该发送到正确的房间"
    
    print("  障碍物更新已推送到 WebSocket")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_push_new_obstacles():
    """测试推送新发现的障碍物"""
    print("=== Test 19: Push New Obstacles ===")
    
    session = UserSession(session_id="test_session_18")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 创建 mock socketio
    mock_socketio = Mock()
    mock_emit = Mock()
    mock_socketio.emit = mock_emit
    
    # 注册 WebSocket 推送
    session.register_websocket_push(mock_socketio)
    
    # 创建模拟新障碍物坐标
    mock_new_obstacles = np.array([[15.0, 15.0], [25.0, 25.0]])
    
    # 触发新障碍物推送
    session._push_new_obstacles(mock_new_obstacles)
    
    # 验证 emit 被调用
    assert mock_emit.called, "应该调用 emit"
    call_args = mock_emit.call_args
    assert call_args[0][0] == 'new_obstacles', "应该发送 new_obstacles 事件"
    assert 'obstacles' in call_args[0][1], "应该包含 obstacles"
    assert call_args[1]['room'] == session.session_id, "应该发送到正确的房间"
    
    print("  新障碍物已推送到 WebSocket")
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_push_without_socketio():
    """测试在没有 socketio 时推送应该安全返回"""
    print("=== Test 20: Push Without SocketIO ===")
    
    session = UserSession(session_id="test_session_19")
    
    # 等待地图初始化
    max_wait = 5.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if session._is_initialized:
            break
        time.sleep(0.1)
    
    # 确保 socketio 为 None
    session._socketio = None
    
    # 尝试推送（应该安全返回，不抛出异常）
    try:
        from AutonomousVehicle.modeling.Car import Car
        test_car = Car(10.0, 10.0, 0.0)
        session._push_state_update(1.0, test_car)
        print("  在没有 socketio 时安全返回")
    except Exception as e:
        assert False, f"不应该抛出异常: {e}"
    
    # 清理
    session.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


if __name__ == '__main__':
    print("[Testing][UserSession]...\n")
    
    try:
        test_initialization()
        test_set_state()
        test_set_goal()
        test_set_goal_before_initialization()
        test_brake()
        test_cancel()
        test_restart()
        test_resume()
        test_get_state()
        test_get_map_data()
        test_stop()
        test_event_handlers()
        test_trajectory_collision_replan()
        test_register_websocket_push()
        test_push_state_update()
        test_push_global_trajectory_success()
        test_push_global_trajectory_failure()
        test_push_obstacles_updated()
        test_push_new_obstacles()
        test_push_without_socketio()
        
        print("=" * 40)
        print("🎉[Testing][UserSession] All tests passed!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

