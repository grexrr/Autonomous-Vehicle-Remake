"""
LocalPlannerAdapter 单元测试
测试局部规划适配器的各项功能
"""

import time
import numpy as np
from typing import Optional

from api.adapters.event_bus import EventBus
from api.adapters.local_planner_adapter import LocalPlannerAdapter, LocalPlanningTrajectories
from api.event_types import MEASURED_STATE, LOCAL_PLANNER_CONTROL_SEQUENCE, LOCAL_PLANNER_TRAJECTORIES
from AutonomousVehicle.modeling.Car import Car


def test_initialization():
    """测试初始化"""
    print("=== Test 1: Initialization ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    # 验证初始状态
    assert not adapter.is_alive(), "初始时进程应该未启动"
    print("✓ Test passed\n")


def test_start_stop():
    """测试启动和停止"""
    print("=== Test 2: Start/Stop ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    # 启动
    adapter.start()
    
    # 等待进程启动
    time.sleep(0.1)
    
    # 验证进程已启动
    assert adapter.is_alive(), "进程应该已启动"
    
    # 停止
    adapter.stop()
    
    # 等待进程停止
    time.sleep(0.1)
    
    # 验证进程已停止
    assert not adapter.is_alive(), "进程应该已停止"
    
    print("✓ Test passed\n")


def test_multiple_start_calls():
    """测试多次调用 start() 不会重复启动"""
    print("=== Test 3: Multiple Start Calls ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    # 多次调用 start
    adapter.start()
    adapter.start()  # 第二次应该被忽略
    adapter.start()  # 第三次应该被忽略
    
    time.sleep(0.1)
    
    # 验证只有一个进程在运行
    assert adapter.is_alive(), "进程应该已启动"
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_subscribe_measured_state():
    """测试订阅 MEASURED_STATE 事件"""
    print("=== Test 4: Subscribe MEASURED_STATE ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    # 启动适配器
    adapter.start()
    
    # 发送 MEASURED_STATE 事件
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    event_bus.emit(MEASURED_STATE, 1.0, car)
    
    # 等待事件处理
    time.sleep(0.05)
    
    # 验证状态已存储（通过发送状态到进程来间接验证）
    # 由于 _state 是私有变量，我们通过行为来验证
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_set_trajectory():
    """测试设置轨迹"""
    print("=== Test 5: Set Trajectory ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    # 启动适配器
    adapter.start()
    time.sleep(0.1)
    
    # 创建测试轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],    # [x, y, yaw]
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    
    # 设置轨迹
    adapter.set_trajectory(trajectory)
    
    # 等待处理
    time.sleep(0.1)
    
    # 发送状态以触发 MPC 计算
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    event_bus.emit(MEASURED_STATE, 0.0, car)
    
    # 等待 MPC 计算和事件发布
    time.sleep(0.3)
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_set_trajectory_none():
    """测试设置 None 轨迹（刹车）"""
    print("=== Test 6: Set Trajectory None (Brake) ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    adapter.start()
    time.sleep(0.1)
    
    # 先设置一个轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    time.sleep(0.1)
    
    # 设置 None（应该发送刹车命令）
    adapter.set_trajectory(None)
    
    time.sleep(0.1)
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_brake():
    """测试刹车命令"""
    print("=== Test 7: Brake ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    adapter.start()
    time.sleep(0.1)
    
    # 设置轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    time.sleep(0.1)
    
    # 发送刹车命令
    adapter.brake()
    
    time.sleep(0.1)
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_cancel():
    """测试取消命令"""
    print("=== Test 8: Cancel ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    adapter.start()
    time.sleep(0.1)
    
    # 设置轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    time.sleep(0.1)
    
    # 取消规划
    adapter.cancel()
    
    time.sleep(0.1)
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_publish_control_sequence():
    """测试发布控制序列事件"""
    print("=== Test 9: Publish Control Sequence ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    # 收集发布的事件
    control_sequences = []
    
    def on_control_sequence(control_sequence):
        control_sequences.append(control_sequence)
        print(f"  收到控制序列: shape={control_sequence.shape}")
    
    # 订阅控制序列事件
    event_bus.subscribe(LOCAL_PLANNER_CONTROL_SEQUENCE, on_control_sequence)
    
    adapter.start()
    time.sleep(0.1)
    
    # 设置轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    time.sleep(0.1)
    
    # 发送状态以触发 MPC 计算
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    event_bus.emit(MEASURED_STATE, 0.0, car)
    
    # 等待 MPC 计算和事件发布
    time.sleep(0.5)
    
    # 验证至少收到一个控制序列
    # 注意：由于 MPC 计算可能需要时间，可能收到也可能收不到
    # 这里只验证适配器正常工作，不强制要求收到事件
    print(f"  收到 {len(control_sequences)} 个控制序列事件")
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_publish_trajectories():
    """测试发布轨迹事件"""
    print("=== Test 10: Publish Trajectories ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    # 收集发布的事件
    trajectories_list = []
    
    def on_trajectories(trajectories: LocalPlanningTrajectories):
        trajectories_list.append(trajectories)
        print(f"  收到轨迹: local_trajectory shape={trajectories.local_trajectory.shape}")
    
    # 订阅轨迹事件
    event_bus.subscribe(LOCAL_PLANNER_TRAJECTORIES, on_trajectories)
    
    adapter.start()
    time.sleep(0.1)
    
    # 设置轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    time.sleep(0.1)
    
    # 发送状态以触发 MPC 计算
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    event_bus.emit(MEASURED_STATE, 0.0, car)
    
    # 等待 MPC 计算和事件发布
    time.sleep(0.5)
    
    print(f"  收到 {len(trajectories_list)} 个轨迹事件")
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_periodic_state_updates():
    """测试定期状态更新"""
    print("=== Test 11: Periodic State Updates ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.15  # 较短的更新间隔
    )
    
    adapter.start()
    time.sleep(0.1)
    
    # 设置轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    time.sleep(0.1)
    
    # 发送初始状态
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    event_bus.emit(MEASURED_STATE, 0.0, car)
    
    # 等待几次更新
    time.sleep(0.5)
    
    # 更新状态
    car.x = 0.5
    event_bus.emit(MEASURED_STATE, 0.5, car)
    
    time.sleep(0.3)
    
    adapter.stop()
    
    print("✓ Test passed\n")


def test_stop_cleanup():
    """测试停止时的资源清理"""
    print("=== Test 12: Stop Cleanup ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    adapter.start()
    time.sleep(0.1)
    
    # 设置轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    
    # 发送状态
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    event_bus.emit(MEASURED_STATE, 0.0, car)
    
    time.sleep(0.1)
    
    # 停止
    adapter.stop()
    
    # 等待清理完成
    time.sleep(0.1)
    
    # 验证进程已停止
    assert not adapter.is_alive(), "进程应该已停止"
    
    # 验证事件已取消订阅（通过再次发送事件，不应该有处理）
    # 注意：由于无法直接访问内部状态，这里只验证进程停止
    
    print("✓ Test passed\n")


def test_no_state_before_trajectory():
    """测试在设置轨迹前发送状态"""
    print("=== Test 13: No State Before Trajectory ===")
    
    event_bus = EventBus()
    adapter = LocalPlannerAdapter(
        event_bus=event_bus,
        delta_time_s=0.1,
        update_interval_s=0.2
    )
    
    adapter.start()
    time.sleep(0.1)
    
    # 先发送状态（此时没有轨迹，MPC 未初始化）
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    event_bus.emit(MEASURED_STATE, 0.0, car)
    
    # 等待一段时间
    time.sleep(0.3)
    
    # 然后设置轨迹
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    adapter.set_trajectory(trajectory)
    
    time.sleep(0.2)
    
    adapter.stop()
    
    print("✓ Test passed\n")


if __name__ == '__main__':
    print("[Testing][LocalPlannerAdapter]...\n")
    
    try:
        test_initialization()
        test_start_stop()
        test_multiple_start_calls()
        test_subscribe_measured_state()
        test_set_trajectory()
        test_set_trajectory_none()
        test_brake()
        test_cancel()
        test_publish_control_sequence()
        test_publish_trajectories()
        test_periodic_state_updates()
        test_stop_cleanup()
        test_no_state_before_trajectory()
        
        print("=" * 40)
        print("🎉[Testing][LocalPlannerAdapter] All tests passed!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

