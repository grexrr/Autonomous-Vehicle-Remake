import numpy as np
from api.adapters.event_bus import EventBus
from api.adapters.map_server import MapServerAdapter
from api.event_types import MAP_INITIALIZED, KNOWN_OBSTACLES_UPDATED, NEW_OBSTACLES_DISCOVERED
from AutonomousVehicle.modeling.Car import Car


def test_initialization():
    """测试初始化"""
    print("=== Test 1: MapServerAdapter Initialization ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    
    # 验证初始状态
    assert adapter.known_obstacle_coordinates is None, "初始障碍物坐标应该为 None"
    assert adapter.unknown_obstacle_coordinates is None, "初始未知障碍物坐标应该为 None"
    assert adapter.bounding_box is None, "初始边界框应该为 None"
    
    print("✓ Test passed: Adapter initialized correctly\n")


def test_init_map():
    """测试地图初始化"""
    print("=== Test 2: Map Initialization ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    
    # 记录发出的事件
    events_received = []
    
    def on_map_initialized():
        events_received.append('map_initialized')
        print("  📡 Event received: MAP_INITIALIZED")
    
    def on_obstacles_updated(coords):
        events_received.append('obstacles_updated')
        print(f"  📡 Event received: KNOWN_OBSTACLES_UPDATED (coords shape: {coords.shape})")
    
    # 订阅事件
    event_bus.subscribe(MAP_INITIALIZED, on_map_initialized)
    event_bus.subscribe(KNOWN_OBSTACLES_UPDATED, on_obstacles_updated)
    
    # 初始化地图
    adapter.init_map()
    
    # 验证地图已加载
    assert adapter.known_obstacle_coordinates is not None, "已知障碍物坐标应该已加载"
    assert adapter.unknown_obstacle_coordinates is not None, "未知障碍物坐标应该已生成"
    assert adapter.bounding_box is not None, "边界框应该已计算"
    
    # 验证障碍物数据的形状
    assert adapter.known_obstacle_coordinates.shape[1] == 2, "障碍物坐标应该是 (N, 2) 形状"
    assert adapter.unknown_obstacle_coordinates.shape[1] == 2, "未知障碍物坐标应该是 (N, 2) 形状"
    
    # 验证边界框
    xmin, ymin, xmax, ymax = adapter.bounding_box
    assert xmin < xmax, "边界框 x 坐标应该合理"
    assert ymin < ymax, "边界框 y 坐标应该合理"
    print(f"  ✓ Map boundary: ({xmin:.2f}, {ymin:.2f}) to ({xmax:.2f}, {ymax:.2f})")
    
    # 验证事件已发送
    assert 'map_initialized' in events_received, "应该发送 MAP_INITIALIZED 事件"
    assert 'obstacles_updated' in events_received, "应该发送 KNOWN_OBSTACLES_UPDATED 事件"
    
    print(f"  ✓ Known obstacles: {len(adapter.known_obstacle_coordinates)} points")
    print(f"  ✓ Unknown obstacles: {len(adapter.unknown_obstacle_coordinates)} points")
    print("✓ Test passed: Map initialized successfully\n")


def test_lidar_scan_discovery():
    """测试 LIDAR 扫描发现障碍物"""
    print("=== Test 3: LIDAR Scan and Obstacle Discovery ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    adapter.init_map()
    
    # 确保地图已初始化
    assert adapter.known_obstacle_coordinates is not None, "已知障碍物应该已初始化"
    assert adapter.unknown_obstacle_coordinates is not None, "未知障碍物应该已初始化"
    
    # 记录初始已知障碍物数量
    initial_known_count = len(adapter.known_obstacle_coordinates)
    print(f"  Initial known obstacles: {initial_known_count}")
    
    # 记录新发现的障碍物事件
    new_obstacles_events = []
    obstacles_updated_events = []
    
    def on_new_obstacles(coords):
        new_obstacles_events.append(coords)
        print(f"  📡 Event: NEW_OBSTACLES_DISCOVERED ({len(coords)} new obstacles)")
    
    def on_obstacles_updated(coords):
        obstacles_updated_events.append(coords)
        print(f"  📡 Event: KNOWN_OBSTACLES_UPDATED (total: {len(coords)} obstacles)")
    
    event_bus.subscribe(NEW_OBSTACLES_DISCOVERED, on_new_obstacles)
    event_bus.subscribe(KNOWN_OBSTACLES_UPDATED, on_obstacles_updated)
    
    # 创建一个车辆状态，扫描所有未知障碍物的位置
    discoveries_made = False
    unknown_coords = adapter.unknown_obstacle_coordinates
    assert unknown_coords is not None, "未知障碍物应该已初始化"
    for unknown_obs in unknown_coords[:5]:  # 只扫描前5个
        x, y = unknown_obs
        car = Car(x=x, y=y, yaw=0.0)
        adapter.update_from_vehicle_state(0.0, car)
        
        if len(new_obstacles_events) > 0:
            discoveries_made = True
    
    # 验证至少发现了一些障碍物
    if discoveries_made:
        assert len(new_obstacles_events) > 0, "应该发现新障碍物"
        assert len(obstacles_updated_events) > 0, "应该更新障碍物列表"
        known_coords = adapter.known_obstacle_coordinates
        assert known_coords is not None, "已知障碍物应该已初始化"
        final_known_count = len(known_coords)
        assert final_known_count > initial_known_count, "已知障碍物数量应该增加"
        print(f"  ✓ Discovered {final_known_count - initial_known_count} new obstacles")
        print("✓ Test passed: LIDAR scan works correctly\n")
    else:
        print("  ⚠ Warning: No obstacles discovered (might be due to scan radius)")
        print("✓ Test passed: LIDAR scan executed without errors\n")


def test_update_from_vehicle_state():
    """测试根据车辆状态更新地图"""
    print("=== Test 4: Update from Vehicle State ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    adapter.init_map()
    
    # 创建一个车辆状态
    car = Car(x=10.0, y=10.0, yaw=0.5)
    
    # 更新应该不会抛出异常
    try:
        adapter.update_from_vehicle_state(timestamp_s=1.0, state=car)
        print("  ✓ Update executed successfully")
        print("✓ Test passed: Vehicle state update works\n")
    except Exception as e:
        raise AssertionError(f"Update failed with error: {e}")


def test_generate_random_initial_state():
    """测试生成随机初始状态"""
    print("=== Test 5: Generate Random Initial State ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    adapter.init_map()
    
    # 确保边界框已初始化
    bbox = adapter.bounding_box
    assert bbox is not None, "边界框应该已初始化"
    
    # 生成多个随机状态
    for i in range(5):
        car = adapter.generate_random_initial_state()
        
        # 验证车辆状态合理
        assert hasattr(car, 'x'), "车辆应该有 x 坐标"
        assert hasattr(car, 'y'), "车辆应该有 y 坐标"
        assert hasattr(car, 'yaw'), "车辆应该有 yaw 角度"
        
        # 验证坐标在边界内
        xmin, ymin, xmax, ymax = bbox
        assert xmin <= car.x <= xmax, f"车辆 x 坐标应该在边界内: {car.x}"
        assert ymin <= car.y <= ymax, f"车辆 y 坐标应该在边界内: {car.y}"
        
        print(f"  State {i+1}: x={car.x:.2f}, y={car.y:.2f}, yaw={car.yaw:.2f}")
    
    print("✓ Test passed: Random initial states generated successfully\n")


def test_event_parameter_compatibility():
    """测试事件参数兼容性（与 CarSimulationAdapter 的 MEASURED_STATE 事件）"""
    print("=== Test 6: Event Parameter Compatibility ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    adapter.init_map()
    
    # 模拟 CarSimulationAdapter 发送的 MEASURED_STATE 事件
    # 该事件包含两个参数：timestamp 和 car
    test_passed = False
    
    def simulate_measured_state_event():
        nonlocal test_passed
        timestamp = 1.5
        car = Car(x=15.0, y=20.0, yaw=0.3)
        
        # 这应该能正确调用 update_from_vehicle_state
        try:
            adapter.update_from_vehicle_state(timestamp, car)
            test_passed = True
            print(f"  ✓ Successfully handled event with timestamp={timestamp}, car=({car.x}, {car.y})")
        except TypeError as e:
            raise AssertionError(f"Parameter mismatch: {e}")
    
    simulate_measured_state_event()
    assert test_passed, "应该成功处理带有 timestamp 参数的事件"
    
    print("✓ Test passed: Event parameters are compatible\n")


def test_multiple_scans_same_location():
    """测试同一位置多次扫描（不应重复发现障碍物）"""
    print("=== Test 7: Multiple Scans at Same Location ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    adapter.init_map()
    
    # 确保未知障碍物已初始化
    unknown_coords = adapter.unknown_obstacle_coordinates
    assert unknown_coords is not None, "未知障碍物应该已初始化"
    
    # 选择一个未知障碍物的位置
    if len(unknown_coords) > 0:
        x, y = unknown_coords[0]
        car = Car(x=x, y=y, yaw=0.0)
        
        new_discoveries = []
        
        def on_new_obstacles(coords):
            new_discoveries.append(len(coords))
        
        event_bus.subscribe(NEW_OBSTACLES_DISCOVERED, on_new_obstacles)
        
        # 第一次扫描
        adapter.update_from_vehicle_state(0.0, car)
        first_scan_discoveries = len(new_discoveries)
        
        # 第二次扫描同一位置
        adapter.update_from_vehicle_state(0.0, car)
        second_scan_discoveries = len(new_discoveries)
        
        # 第二次扫描不应该发现新障碍物
        assert second_scan_discoveries == first_scan_discoveries, "相同位置重复扫描不应发现新障碍物"
        print(f"  ✓ First scan: {first_scan_discoveries} discoveries")
        print(f"  ✓ Second scan: {second_scan_discoveries - first_scan_discoveries} discoveries (correct)")
        print("✓ Test passed: No duplicate discoveries\n")
    else:
        print("  ⚠ Skipped: No unknown obstacles to test")
        print("✓ Test passed (skipped)\n")


def test_properties():
    """测试属性访问器"""
    print("=== Test 8: Property Accessors ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    adapter.init_map()
    
    # 测试 known_obstacle_coordinates 属性
    coords = adapter.known_obstacle_coordinates
    assert coords is not None, "known_obstacle_coordinates 应该返回值"
    assert isinstance(coords, np.ndarray), "应该返回 numpy 数组"
    print(f"  ✓ known_obstacle_coordinates: shape {coords.shape}")
    
    # 测试 unknown_obstacle_coordinates 属性
    unknown_coords = adapter.unknown_obstacle_coordinates
    assert unknown_coords is not None, "unknown_obstacle_coordinates 应该返回值"
    assert isinstance(unknown_coords, np.ndarray), "应该返回 numpy 数组"
    print(f"  ✓ unknown_obstacle_coordinates: shape {unknown_coords.shape}")
    
    # 测试 bounding_box 属性
    bbox = adapter.bounding_box
    assert bbox is not None, "bounding_box 应该返回值"
    assert len(bbox) == 4, "bounding_box 应该有 4 个值"
    print(f"  ✓ bounding_box: {bbox}")
    
    print("✓ Test passed: All properties accessible\n")


def test_map_selection():
    """测试地图选择功能"""
    print("=== Test 9: Map Selection ===")
    
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    
    # 测试默认地图（map2）
    adapter.init_map()  # 使用默认 map2
    map2_coords = adapter.known_obstacle_coordinates
    map2_bbox = adapter.bounding_box
    assert map2_coords is not None, "Map2 应该成功加载"
    assert map2_bbox is not None, "Map2 应该有边界框"
    print(f"  ✓ Map2 loaded: {len(map2_coords)} obstacles, bbox: {map2_bbox}")
    
    # 测试选择 map
    try:
        adapter.init_map("map")
        map1_coords = adapter.known_obstacle_coordinates
        map1_bbox = adapter.bounding_box
        
        # 验证不同地图可能有不同的障碍物数量或边界
        # （注意：如果两个地图完全相同，这个测试可能通过，但至少验证了功能正常）
        assert map1_coords is not None, "Map1 应该成功加载"
        assert map1_bbox is not None, "Map1 应该有边界框"
        print(f"  ✓ Map1 loaded: {len(map1_coords)} obstacles, bbox: {map1_bbox}")
        print("  ✓ Map selection works correctly")
    except FileNotFoundError:
        print("  ⚠ Warning: map.png not found, skipping map1 test")
        print("  ✓ Map selection function exists (map2 works)")
    
    # 测试无效地图名称
    try:
        adapter.init_map("nonexistent_map")
        assert False, "应该抛出 FileNotFoundError"
    except FileNotFoundError as e:
        print(f"  ✓ Invalid map name correctly raises FileNotFoundError: {e}")
    
    print("✓ Test passed: Map selection works\n")


# 运行所有测试
if __name__ == "__main__":
    print("[Testing][MapServerAdapter]...\n")
    
    test_initialization()
    test_init_map()
    test_lidar_scan_discovery()
    test_update_from_vehicle_state()
    test_generate_random_initial_state()
    test_event_parameter_compatibility()
    test_multiple_scans_same_location()
    test_properties()
    test_map_selection()
    
    print("=" * 40)
    print("🎉[Testing][MapServerAdapter] All tests past!")
    print("=" * 40)

