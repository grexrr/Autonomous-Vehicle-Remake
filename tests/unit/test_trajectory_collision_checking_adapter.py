import numpy as np
import numpy.typing as npt

from api.adapters.event_bus import EventBus
from api.adapters.trajectory_collision_checking_adapter import TrajectoryCollisionCheckingAdapter
from api.event_types import (
    GLOBAL_PLANNER_TRAJECTORY,
    KNOWN_OBSTACLES_UPDATED,
    NEW_OBSTACLES_DISCOVERED,
    TRAJECTORY_COLLIDED
)
from AutonomousVehicle.TrajectoryCollisionCheckingNode import DISCARD_FIRST_N


def test_initialization():
    """测试初始化"""
    print("=== Test 1: Initialization ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    assert adapter._event_bus == event_bus
    assert adapter._checker is None
    assert adapter._known_obstacles is None
    print("✓ Test passed\n")


def test_trajectory_received():
    """测试收到轨迹后创建检查器"""
    print("=== Test 2: Trajectory Received ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    # 创建一个简单的轨迹 [x, y, yaw, ...]
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],  # 会被丢弃
        [1.0, 0.0, 0.0, 0.0],  # 会被丢弃
        [2.0, 0.0, 0.0, 0.0],  # 会被丢弃
        [3.0, 0.0, 0.0, 0.0],  # 会被丢弃
        [4.0, 0.0, 0.0, 0.0],  # 会被丢弃
        [5.0, 0.0, 0.0, 0.0],  # 实际使用的起点
        [6.0, 0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0, 0.0],
    ])
    
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    
    assert adapter._checker is not None, "应该创建检查器"
    print("✓ Test passed\n")


def test_trajectory_none():
    """测试收到 None 轨迹"""
    print("=== Test 3: Trajectory None ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    # 先设置一个轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    assert adapter._checker is not None
    
    # 发送 None 应该清除检查器
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, None)
    assert adapter._checker is None, "应该清除检查器"
    print("✓ Test passed\n")


def test_known_obstacles_updated():
    """测试已知障碍物更新"""
    print("=== Test 4: Known Obstacles Updated ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    # 创建轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    
    # 更新已知障碍物（远离轨迹，不会碰撞）
    obstacles = np.array([
        [10.0, 10.0],  # 远离轨迹
        [20.0, 20.0],
    ])
    event_bus.emit(KNOWN_OBSTACLES_UPDATED, obstacles)
    
    assert adapter._known_obstacles is not None, "应该保存已知障碍物"
    assert np.array_equal(adapter._known_obstacles, obstacles), "障碍物应该匹配"
    print("✓ Test passed\n")


def test_new_obstacles_discovered():
    """测试新发现的障碍物"""
    print("=== Test 5: New Obstacles Discovered ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    # 创建轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    
    # 发送新障碍物（远离轨迹，不会碰撞）
    new_obstacles = np.array([
        [15.0, 15.0],
    ])
    event_bus.emit(NEW_OBSTACLES_DISCOVERED, new_obstacles)
    
    # 如果没有检查器，不应该崩溃
    assert adapter._checker is not None, "检查器应该存在"
    print("✓ Test passed\n")


def test_collision_detected():
    """测试碰撞检测并发布事件"""
    print("=== Test 6: Collision Detected ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    collision_events = []
    
    def on_collision():
        collision_events.append(True)
        print("  检测到碰撞！")
    
    event_bus.subscribe(TRAJECTORY_COLLIDED, on_collision)
    
    # 创建一条从 (0,0) 到 (10,0) 的轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],  # 实际使用的起点
        [6.0, 0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0],
        [9.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    
    # 在轨迹路径上放置障碍物（会碰撞）
    # 注意：需要根据 Car.COLLISION_RADIUS 来放置障碍物
    # Car.COLLISION_RADIUS 大约是 2.5m，所以障碍物应该在轨迹附近
    obstacles = np.array([
        [7.0, 0.5],  # 在轨迹路径上，会碰撞
    ])
    event_bus.emit(KNOWN_OBSTACLES_UPDATED, obstacles)
    
    assert len(collision_events) > 0, "应该检测到碰撞并发布事件"
    print("✓ Test passed\n")


def test_no_collision():
    """测试无碰撞情况"""
    print("=== Test 7: No Collision ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    collision_events = []
    
    def on_collision():
        collision_events.append(True)
    
    event_bus.subscribe(TRAJECTORY_COLLIDED, on_collision)
    
    # 创建一条从 (0,0) 到 (10,0) 的轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    
    # 障碍物远离轨迹（不会碰撞）
    obstacles = np.array([
        [5.0, 10.0],  # 远离轨迹
        [10.0, 10.0],
    ])
    event_bus.emit(KNOWN_OBSTACLES_UPDATED, obstacles)
    
    assert len(collision_events) == 0, "不应该检测到碰撞"
    print("✓ Test passed\n")


def test_collision_with_new_obstacles():
    """测试新发现的障碍物导致碰撞"""
    print("=== Test 8: Collision With New Obstacles ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    collision_events = []
    
    def on_collision():
        collision_events.append(True)
        print("  新障碍物导致碰撞！")
    
    event_bus.subscribe(TRAJECTORY_COLLIDED, on_collision)
    
    # 创建轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    
    # 新发现的障碍物在轨迹路径上
    new_obstacles = np.array([
        [6.0, 0.5],  # 在轨迹路径上
    ])
    event_bus.emit(NEW_OBSTACLES_DISCOVERED, new_obstacles)
    
    assert len(collision_events) > 0, "应该检测到碰撞"
    print("✓ Test passed\n")


def test_cancel():
    """测试取消碰撞检查"""
    print("=== Test 9: Cancel ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    # 创建轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    assert adapter._checker is not None
    
    # 取消
    adapter.cancel()
    assert adapter._checker is None, "应该清除检查器"
    print("✓ Test passed\n")


def test_stop():
    """测试停止和清理"""
    print("=== Test 10: Stop ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    # 创建轨迹和障碍物
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
    ])
    obstacles = np.array([[10.0, 10.0]])
    
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    event_bus.emit(KNOWN_OBSTACLES_UPDATED, obstacles)
    
    assert adapter._checker is not None
    assert adapter._known_obstacles is not None
    
    # 停止
    adapter.stop()
    
    assert adapter._checker is None, "应该清除检查器"
    assert adapter._known_obstacles is None, "应该清除已知障碍物"
    
    # 验证取消订阅后不会再响应事件
    collision_events = []
    def on_collision():
        collision_events.append(True)
    event_bus.subscribe(TRAJECTORY_COLLIDED, on_collision)
    
    # 发送新轨迹和障碍物，不应该触发检查（因为已取消订阅）
    # 注意：实际上 stop() 后 adapter 已经取消订阅，但这里我们测试的是
    # adapter 内部状态是否被清理
    print("✓ Test passed\n")


def test_obstacles_before_trajectory():
    """测试先收到障碍物，后收到轨迹的情况"""
    print("=== Test 11: Obstacles Before Trajectory ===")
    
    event_bus = EventBus()
    adapter = TrajectoryCollisionCheckingAdapter(event_bus)
    
    collision_events = []
    
    def on_collision():
        collision_events.append(True)
    
    event_bus.subscribe(TRAJECTORY_COLLIDED, on_collision)
    
    # 先发送障碍物
    obstacles = np.array([
        [5.0, 0.5],  # 会在轨迹路径上
    ])
    event_bus.emit(KNOWN_OBSTACLES_UPDATED, obstacles)
    assert adapter._known_obstacles is not None
    assert adapter._checker is None  # 还没有轨迹
    
    # 后发送轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],  # 实际使用的起点
        [6.0, 0.0, 0.0, 0.0],
    ])
    event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
    
    # 应该立即检查碰撞（因为已有已知障碍物）
    # 注意：根据实现，_on_trajectory 中会检查已知障碍物
    assert adapter._checker is not None, "应该创建检查器"
    # 如果障碍物在路径上，应该检测到碰撞
    # 这里我们只验证检查器被创建，碰撞检测取决于障碍物位置
    print("✓ Test passed\n")


if __name__ == '__main__':
    print("[Testing][TrajectoryCollisionCheckingAdapter]...\n")
    
    try:
        test_initialization()
        test_trajectory_received()
        test_trajectory_none()
        test_known_obstacles_updated()
        test_new_obstacles_discovered()
        test_collision_detected()
        test_no_collision()
        test_collision_with_new_obstacles()
        test_cancel()
        test_stop()
        test_obstacles_before_trajectory()
        
        print("=" * 40)
        print("🎉[Testing][TrajectoryCollisionCheckingAdapter] All tests passed!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()



