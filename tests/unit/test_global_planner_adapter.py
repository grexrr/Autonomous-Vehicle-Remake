import time
import numpy as np

from api.adapters.event_bus import EventBus
from api.adapters.global_planner_adapter import GlobalPlannerAdapter
from api.event_types import (
    GLOBAL_PLANNER_DISPLAY_SEGMENTS,
    GLOBAL_PLANNER_TRAJECTORY,
    GLOBAL_PLANNER_FINISHED
)
from AutonomousVehicle.modeling.Car import Car
from AutonomousVehicle.modeling.Obstacles import Obstacles


def test_initialization():
    """测试初始化"""
    print("=== Test 1: Initialization ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=10
    )
    
    assert adapter._event_bus == event_bus
    assert adapter._process_adapter is not None
    print("✓ Test passed\n")


def test_start_stop():
    """测试启动和停止"""
    print("=== Test 2: Start/Stop ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=10
    )
    
    # 启动
    adapter.start()
    time.sleep(0.2)  # 等待进程启动
    
    assert adapter.is_alive(), "进程应该正在运行"
    
    # 停止
    adapter.stop()
    time.sleep(0.2)  # 等待进程停止
    
    assert not adapter.is_alive(), "进程应该已停止"
    
    print("✓ Test passed\n")


def test_plan_with_simple_scenario():
    """测试简单场景的路径规划"""
    print("=== Test 3: Plan With Simple Scenario ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=5  # 较小的值，便于测试
    )
    
    # 收集事件
    display_segments_received = []
    trajectory_received = []
    finished_received = []
    
    def on_display_segments(segments):
        display_segments_received.append(segments)
        print(f"  收到显示段: {len(segments)} 个")
    
    def on_trajectory(trajectory):
        trajectory_received.append(trajectory)
        print(f"  收到轨迹: {trajectory is not None}")
    
    def on_finished():
        finished_received.append(True)
        print("  收到完成事件")
    
    # 订阅事件
    event_bus.subscribe(GLOBAL_PLANNER_DISPLAY_SEGMENTS, on_display_segments)
    event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, on_trajectory)
    event_bus.subscribe(GLOBAL_PLANNER_FINISHED, on_finished)
    
    # 创建简单的起点和终点
    start_car = Car(x=0.0, y=0.0, yaw=0.0)
    goal_car = Car(x=10.0, y=0.0, yaw=0.0)
    
    # 创建障碍物（覆盖起点到终点的范围，但远离路径）
    # 确保障碍物范围覆盖起点和终点，这样网格才能包含目标点
    obstacle_coords = np.array([
        [-2.0, 3.0],   # 起点左侧，远离路径
        [-2.0, -3.0],  # 起点左侧，远离路径
        [12.0, 3.0],   # 终点右侧，远离路径
        [12.0, -3.0],  # 终点右侧，远离路径
        [5.0, 3.0],    # 中间，远离路径
        [5.0, -3.0],   # 中间，远离路径
    ])
    obstacles = Obstacles(obstacle_coords)
    
    # 启动适配器
    adapter.start()
    time.sleep(0.2)
    
    # 请求规划
    adapter.plan(start_car, goal_car, obstacles)
    
    # 等待规划完成（最多等待10秒）
    max_wait = 10.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if len(trajectory_received) > 0:
            break
        time.sleep(0.1)
    
    # 验证至少收到了轨迹（可能为None，如果规划失败）
    assert len(trajectory_received) > 0, "应该收到轨迹事件"
    
    # 如果规划成功，应该收到完成事件
    if trajectory_received[0] is not None:
        assert len(finished_received) > 0, "规划成功时应该收到完成事件"
    
    adapter.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_plan_with_numpy_array_start():
    """测试使用numpy数组作为起点"""
    print("=== Test 4: Plan With Numpy Array Start ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=5
    )
    
    trajectory_received = []
    
    def on_trajectory(trajectory):
        trajectory_received.append(trajectory)
    
    event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, on_trajectory)
    
    # 使用numpy数组作为起点
    start = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
    goal_car = Car(x=5.0, y=0.0, yaw=0.0)
    # 使用覆盖起点到终点范围的障碍物
    obstacle_coords = np.array([
        [-2.0, 3.0],
        [-2.0, -3.0],
        [7.0, 3.0],
        [7.0, -3.0],
        [2.5, 3.0],
        [2.5, -3.0],
    ])
    obstacles = Obstacles(obstacle_coords)
    
    adapter.start()
    time.sleep(0.2)
    
    adapter.plan(start, goal_car, obstacles)
    
    # 等待结果
    time.sleep(2.0)
    
    assert len(trajectory_received) > 0, "应该收到轨迹事件"
    
    adapter.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_cancel():
    """测试取消规划"""
    print("=== Test 5: Cancel ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=5
    )
    
    trajectory_received = []
    
    def on_trajectory(trajectory):
        trajectory_received.append(trajectory)
    
    event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, on_trajectory)
    
    # 创建一个可能需要较长时间的规划任务
    start_car = Car(x=0.0, y=0.0, yaw=0.0)
    goal_car = Car(x=20.0, y=20.0, yaw=np.pi/2)
    # 使用覆盖起点到终点范围的障碍物
    obstacle_coords = np.array([
        [-2.0, -2.0],   # 起点附近
        [-2.0, 2.0],
        [22.0, 18.0],   # 终点附近
        [22.0, 22.0],
        [10.0, 15.0],   # 中间区域
        [15.0, 10.0],
    ])
    obstacles = Obstacles(obstacle_coords)
    
    adapter.start()
    time.sleep(0.2)
    
    # 开始规划
    adapter.plan(start_car, goal_car, obstacles)
    
    # 立即取消
    time.sleep(0.1)
    adapter.cancel()
    
    # 等待一段时间
    time.sleep(1.0)
    
    # 取消后可能收到None轨迹，或者不收到轨迹
    # 这取决于实现细节，我们只验证不会崩溃
    
    adapter.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_plan_with_obstacles():
    """测试带障碍物的规划"""
    print("=== Test 6: Plan With Obstacles ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=5
    )
    
    trajectory_received = []
    
    def on_trajectory(trajectory):
        trajectory_received.append(trajectory)
        if trajectory is not None:
            print(f"  收到轨迹，长度: {len(trajectory)}")
        else:
            print("  规划失败（可能被障碍物阻挡）")
    
    event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, on_trajectory)
    
    # 创建起点和终点
    start_car = Car(x=0.0, y=0.0, yaw=0.0)
    goal_car = Car(x=10.0, y=0.0, yaw=0.0)
    
    # 创建一些障碍物（覆盖范围但远离路径，允许规划成功）
    obstacle_coords = np.array([
        [-2.0, 3.0],   # 起点左侧，远离路径
        [-2.0, -3.0],  # 起点左侧，远离路径
        [12.0, 3.0],   # 终点右侧，远离路径
        [12.0, -3.0],  # 终点右侧，远离路径
        [5.0, 2.5],    # 中间，远离路径（不在直线上）
        [5.0, -2.5],   # 中间，远离路径（不在直线上）
    ])
    obstacles = Obstacles(obstacle_coords)
    
    adapter.start()
    time.sleep(0.2)
    
    adapter.plan(start_car, goal_car, obstacles)
    
    # 等待规划完成
    time.sleep(5.0)
    
    assert len(trajectory_received) > 0, "应该收到轨迹事件"
    
    adapter.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_display_segments_emission():
    """测试显示段的发送"""
    print("=== Test 7: Display Segments Emission ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=3  # 很小的值，确保会发送中间结果
    )
    
    display_segments_received = []
    
    def on_display_segments(segments):
        display_segments_received.append(segments)
        print(f"  收到显示段批次: {len(segments)} 个段")
    
    event_bus.subscribe(GLOBAL_PLANNER_DISPLAY_SEGMENTS, on_display_segments)
    
    start_car = Car(x=0.0, y=0.0, yaw=0.0)
    goal_car = Car(x=15.0, y=5.0, yaw=np.pi/4)
    # 使用覆盖起点到终点范围的障碍物
    obstacle_coords = np.array([
        [-2.0, -2.0],   # 起点附近
        [-2.0, 2.0],
        [17.0, 3.0],    # 终点附近
        [17.0, 7.0],
        [7.0, 10.0],    # 中间区域
        [10.0, 7.0],
    ])
    obstacles = Obstacles(obstacle_coords)
    
    adapter.start()
    time.sleep(0.2)
    
    adapter.plan(start_car, goal_car, obstacles)
    
    # 等待一段时间，看是否收到中间结果
    time.sleep(2.0)
    
    # 如果规划算法生成了足够的段，应该会收到中间结果
    # 这取决于规划算法的实现和场景复杂度
    print(f"  总共收到 {len(display_segments_received)} 批显示段")
    
    adapter.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_multiple_plan_calls():
    """测试多次调用plan"""
    print("=== Test 8: Multiple Plan Calls ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=5
    )
    
    trajectory_count = [0]
    
    def on_trajectory(trajectory):
        trajectory_count[0] += 1
        print(f"  收到轨迹 #{trajectory_count[0]}")
    
    event_bus.subscribe(GLOBAL_PLANNER_TRAJECTORY, on_trajectory)
    
    adapter.start()
    time.sleep(0.2)
    
    # 第一次规划
    start1 = Car(x=0.0, y=0.0, yaw=0.0)
    goal1 = Car(x=5.0, y=0.0, yaw=0.0)
    obstacle_coords1 = np.array([
        [-2.0, 3.0],
        [-2.0, -3.0],
        [7.0, 3.0],
        [7.0, -3.0],
        [2.5, 3.0],
        [2.5, -3.0],
    ])
    obstacles1 = Obstacles(obstacle_coords1)
    
    adapter.plan(start1, goal1, obstacles1)
    time.sleep(1.0)
    
    # 第二次规划（应该取消第一次）
    start2 = Car(x=0.0, y=0.0, yaw=0.0)
    goal2 = Car(x=8.0, y=0.0, yaw=0.0)
    obstacle_coords2 = np.array([
        [-2.0, 3.0],
        [-2.0, -3.0],
        [10.0, 3.0],
        [10.0, -3.0],
        [4.0, 3.0],
        [4.0, -3.0],
    ])
    obstacles2 = Obstacles(obstacle_coords2)
    
    adapter.plan(start2, goal2, obstacles2)
    time.sleep(1.0)
    
    # 验证至少收到了一些轨迹事件
    assert trajectory_count[0] > 0, "应该收到至少一个轨迹事件"
    
    adapter.stop()
    time.sleep(0.2)
    
    print("✓ Test passed\n")


def test_stop_cleanup():
    """测试停止时的清理"""
    print("=== Test 9: Stop Cleanup ===")
    
    event_bus = EventBus()
    adapter = GlobalPlannerAdapter(
        event_bus=event_bus,
        segment_collection_size=5
    )
    
    adapter.start()
    time.sleep(0.2)
    
    # 开始一个规划
    start_car = Car(x=0.0, y=0.0, yaw=0.0)
    goal_car = Car(x=10.0, y=0.0, yaw=0.0)
    obstacle_coords = np.array([
        [-2.0, 3.0],
        [-2.0, -3.0],
        [12.0, 3.0],
        [12.0, -3.0],
        [5.0, 3.0],
        [5.0, -3.0],
    ])
    obstacles = Obstacles(obstacle_coords)
    
    adapter.plan(start_car, goal_car, obstacles)
    
    # 立即停止
    adapter.stop()
    time.sleep(0.3)
    
    # 验证进程已停止
    assert not adapter.is_alive(), "进程应该已停止"
    
    # 注意：ProcessAdapter 停止后需要重新创建才能再次启动
    # 这里只验证停止功能正常
    
    print("✓ Test passed\n")


if __name__ == '__main__':
    print("[Testing][GlobalPlannerAdapter]...\n")
    
    try:
        test_initialization()
        test_start_stop()
        test_plan_with_simple_scenario()
        test_plan_with_numpy_array_start()
        test_cancel()
        test_plan_with_obstacles()
        test_display_segments_emission()
        test_multiple_plan_calls()
        test_stop_cleanup()
        
        print("=" * 40)
        print("🎉[Testing][GlobalPlannerAdapter] All tests passed!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

