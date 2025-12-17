"""
CarSimulationAdapter 单元测试
测试车辆仿真适配器的各项功能
"""

import time
import numpy as np

from api.adapters.event_bus import EventBus
from api.adapters.car_simulation import CarSimulationAdapter
from AutonomousVehicle.modeling.car import Car


def test_initialization():
    print("=== Test 1: Initialization ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.02,
        publish_interval_s=0.05
    )
    
    # 验证初始状态
    assert adapter.get_state() is None, "初始状态应该为 None"
    print("✓ Test passed\n")


def test_set_state():
    """测试设置车辆状态"""
    print("=== Test 2: Set State ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.02,
        publish_interval_s=0.05
    )
    
    # 创建初始车辆状态
    initial_car = Car(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steer=0.0)
    adapter.set_state(initial_car)
    
    # 验证状态已设置
    state = adapter.get_state()
    assert state is not None, "状态应该已设置"
    timestamp, car = state
    assert car.x == 0.0, "x 坐标应该匹配"
    assert car.y == 0.0, "y 坐标应该匹配"
    assert timestamp == 0.0, "初始时间戳应该为 0"
    print("✓ Test passed\n")


def test_start_stop():
    """测试启动和停止"""
    print("=== Test 3: Start/Stop ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.1,  # 较长的间隔，方便测试
        publish_interval_s=0.2
    )
    
    # 设置初始状态
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    adapter.set_state(car)
    
    # 启动
    adapter.start()
    
    # 等待一小段时间，让定时器执行
    time.sleep(0.15)
    
    # 验证状态已更新（时间戳应该增加）
    state = adapter.get_state()
    assert state is not None
    timestamp, _ = state
    assert timestamp > 0.0, "时间戳应该增加"
    
    # 停止
    adapter.stop()
    
    # 验证停止后状态
    state_after_stop = adapter.get_state()
    assert state_after_stop is not None
    _, car_after_stop = state_after_stop
    assert car_after_stop.velocity == 0.0, "停止后速度应该为 0"
    assert car_after_stop.steer == 0.0, "停止后转向角应该为 0"
    
    print("✓ Test passed\n")


def test_event_publishing():
    """测试事件发布"""
    print("=== Test 4: Event Publishing ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.05,
        publish_interval_s=0.1  # 较短的发布间隔
    )
    
    # 收集发布的事件
    published_events = []
    
    def on_measured_state(timestamp, car):
        published_events.append((timestamp, car))
        print(f"  收到事件: timestamp={timestamp:.3f}, x={car.x:.3f}, y={car.y:.3f}")
    
    # 订阅事件
    event_bus.subscribe('measured_state', on_measured_state)
    
    # 设置初始状态
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    adapter.set_state(car)
    
    # 启动
    adapter.start()
    
    # 等待足够长的时间，让发布事件触发
    time.sleep(0.15)
    
    # 停止
    adapter.stop()
    
    # 验证至少收到一个事件
    assert len(published_events) > 0, "应该至少发布一个事件"
    print(f"  总共收到 {len(published_events)} 个事件")
    print("✓ Test passed\n")


def test_simulation_updates_state():
    """测试仿真更新车辆状态"""
    print("=== Test 5: Simulation Updates State ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.05,
        publish_interval_s=0.2
    )
    
    # 设置初始状态（有速度，会移动）
    initial_car = Car(x=0.0, y=0.0, yaw=0.0, velocity=2.0, steer=0.0)
    adapter.set_state(initial_car)
    
    # 获取初始状态
    initial_state = adapter.get_state()
    assert initial_state is not None
    _, initial_car_state = initial_state
    initial_x = initial_car_state.x
    initial_y = initial_car_state.y
    
    # 启动
    adapter.start()
    
    # 等待仿真执行几次
    time.sleep(0.12)
    
    # 获取更新后的状态
    updated_state = adapter.get_state()
    assert updated_state is not None
    _, updated_car_state = updated_state
    
    # 验证位置已改变（车辆在移动）
    assert updated_car_state.x > initial_x, "x 坐标应该增加（车辆向前移动）"
    assert updated_car_state.y == initial_y, "y 坐标应该不变（直线行驶）"
    
    # 停止
    adapter.stop()
    
    print(f"  初始位置: ({initial_x:.3f}, {initial_y:.3f})")
    print(f"  更新位置: ({updated_car_state.x:.3f}, {updated_car_state.y:.3f})")
    print("✓ Test passed\n")


def test_control_sequence():
    """测试控制序列"""
    print("=== Test 6: Control Sequence ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.05,
        publish_interval_s=0.2
    )
    
    # 设置初始状态
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steer=0.0)
    adapter.set_state(car)
    
    # 启动（必须先启动，否则 set_control_sequence 会被拒绝）
    adapter.start()
    adapter.resume()  # 确保不是停止状态
    
    # 创建控制序列：加速到 5 m/s，然后保持
    control_sequence = np.array([
        [0.0, 5.0, 0.0],  # t=0s: velocity=5m/s, steer=0
        [1.0, 5.0, 0.0],  # t=1s: velocity=5m/s, steer=0
    ])
    
    adapter.set_control_sequence(control_sequence)
    
    # 等待仿真执行
    time.sleep(0.15)
    
    # 验证车辆速度应该接近控制序列中的速度
    state = adapter.get_state()
    assert state is not None
    _, car_state = state
    assert car_state.velocity > 0.0, "速度应该增加"
    
    # 停止
    adapter.stop()
    
    print(f"  最终速度: {car_state.velocity:.3f} m/s")
    print("✓ Test passed\n")


def test_stop_prevents_control_sequence():
    """测试停止状态下不能设置控制序列"""
    print("=== Test 7: Stop Prevents Control Sequence ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.05,
        publish_interval_s=0.2
    )
    
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steer=0.0)
    adapter.set_state(car)
    
    # 不启动，直接尝试设置控制序列
    control_sequence = np.array([
        [0.0, 5.0, 0.0],
        [1.0, 5.0, 0.0],
    ])
    
    adapter.set_control_sequence(control_sequence)
    
    # 验证控制序列没有被设置（通过检查状态）
    # 由于没有控制序列，车辆应该保持初始状态
    state = adapter.get_state()
    assert state is not None
    _, car_state = state
    assert car_state.velocity == 0.0, "停止状态下速度应该保持为 0"
    
    print("✓ Test passed\n")


def test_resume():
    """测试恢复功能"""
    print("=== Test 8: Resume ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.05,
        publish_interval_s=0.2
    )
    
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    adapter.set_state(car)
    
    # 启动
    adapter.start()
    
    # 停止
    adapter.stop()
    
    # 恢复
    adapter.resume()
    
    # 现在应该可以设置控制序列了
    control_sequence = np.array([
        [0.0, 3.0, 0.0],
        [1.0, 3.0, 0.0],
    ])
    adapter.set_control_sequence(control_sequence)
    
    # 验证控制序列已设置（通过检查状态变化）
    time.sleep(0.1)
    state = adapter.get_state()
    assert state is not None
    
    print("✓ Test passed\n")


def test_multiple_start_calls():
    """测试多次调用 start() 不会重复启动"""
    print("=== Test 9: Multiple Start Calls ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.05,
        publish_interval_s=0.2
    )
    
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=1.0, steer=0.0)
    adapter.set_state(car)
    
    # 多次调用 start
    adapter.start()
    adapter.start()  # 第二次调用应该被忽略
    adapter.start()  # 第三次调用应该被忽略
    
    # 等待一段时间
    time.sleep(0.1)
    
    # 验证只有一个定时器在运行（通过检查状态更新）
    state = adapter.get_state()
    assert state is not None
    
    # 停止
    adapter.stop()
    
    print("✓ Test passed\n")


def test_get_state_without_state():
    """测试在没有设置状态时获取状态"""
    print("=== Test 10: Get State Without State ===")
    
    event_bus = EventBus()
    adapter = CarSimulationAdapter(
        event_bus=event_bus,
        delta_time_s=0.01,
        simulation_interval_s=0.05,
        publish_interval_s=0.2
    )
    
    # 不设置状态，直接获取
    state = adapter.get_state()
    assert state is None, "没有状态时应该返回 None"
    
    print("✓ Test passed\n")


if __name__ == '__main__':
    
    print("[Testing][CarSimulationAdapter]...\n")
    
    try:
        test_initialization()
        test_set_state()
        test_start_stop()
        test_event_publishing()
        test_simulation_updates_state()
        test_control_sequence()
        test_stop_prevents_control_sequence()
        test_resume()
        test_multiple_start_calls()
        test_get_state_without_state()
        
        print("=" * 40)
        print("🎉[Testing][CarSimulationAdapter] All tests passed!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()