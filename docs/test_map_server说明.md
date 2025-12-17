# MapServerAdapter 测试说明

## 测试文件
`test_map_server.py` - MapServerAdapter 的完整单元测试套件

## 测试覆盖范围

### 1. 初始化测试 (`test_initialization`)
- 验证适配器的初始状态
- 确保所有属性正确初始化为 None

### 2. 地图初始化测试 (`test_init_map`)
- 测试地图数据加载
- 验证事件发布（MAP_INITIALIZED, KNOWN_OBSTACLES_UPDATED）
- 检查障碍物数据格式和边界框计算

### 3. LIDAR 扫描测试 (`test_lidar_scan_discovery`)
- 测试动态障碍物发现
- 验证 NEW_OBSTACLES_DISCOVERED 事件
- 确认已知障碍物列表更新

### 4. 车辆状态更新测试 (`test_update_from_vehicle_state`)
- 测试根据车辆位置更新地图
- 验证方法调用不抛出异常

### 5. 随机初始状态生成测试 (`test_generate_random_initial_state`)
- 测试生成无碰撞的随机车辆位置
- 验证生成的位置在地图边界内

### 6. 事件参数兼容性测试 (`test_event_parameter_compatibility`)
- **重要！** 验证与 CarSimulationAdapter 的 MEASURED_STATE 事件兼容
- 确保 `update_from_vehicle_state(timestamp, car)` 接受两个参数

### 7. 重复扫描测试 (`test_multiple_scans_same_location`)
- 验证同一位置多次扫描不会重复发现障碍物
- 测试已发现标记逻辑

### 8. 属性访问测试 (`test_properties`)
- 测试所有属性访问器
- 验证返回值类型

## 运行测试

### 方法 1: 使用 pytest
```bash
cd /Users/grexrr/Documents/Autonomous-Vehicle-Remake
python -m pytest tests/unit/test_map_server.py -v
```

### 方法 2: 直接运行
```bash
cd /Users/grexrr/Documents/Autonomous-Vehicle-Remake
python tests/unit/test_map_server.py
```

### 方法 3: 在 Python 环境中运行
```python
import sys
sys.path.insert(0, '/Users/grexrr/Documents/Autonomous-Vehicle-Remake')

from tests.unit.test_map_server import *

# 运行单个测试
test_initialization()
test_init_map()

# 或运行全部
import tests.unit.test_map_server as test_module
if hasattr(test_module, '__main__'):
    test_module.__main__()
```

## 已知问题

### Apple Silicon (M2) Segmentation Fault
在某些 Apple Silicon 环境中，特别是使用 Anaconda 时，可能会遇到段错误 (Exit code: 139)。

**原因：**
- numpy 与 Apple Silicon 的兼容性问题
- scipy/cv2 在特定环境下的问题

**解决方案：**

#### 选项 1: 使用系统 Python
```bash
/usr/bin/python3 tests/unit/test_map_server.py
```

#### 选项 2: 重新安装 numpy（针对 ARM64）
```bash
conda uninstall numpy scipy
pip install numpy scipy --force-reinstall --no-binary :all:
```

#### 选项 3: 使用 miniforge (推荐)
```bash
# 安装 miniforge（原生支持 ARM64）
brew install miniforge
conda create -n av_test python=3.11
conda activate av_test
pip install -r requirements.txt
```

#### 选项 4: 在 Docker 中运行测试
```bash
docker run -v $(pwd):/app python:3.11-slim bash -c "cd /app && pip install -r requirements.txt && python tests/unit/test_map_server.py"
```

## 测试最佳实践

### 编写新测试时注意：

1. **独立性** - 每个测试应该独立运行，不依赖其他测试
2. **可重复性** - 多次运行应该得到相同结果
3. **清晰性** - 使用描述性的测试名称和注释
4. **覆盖边界情况** - 测试正常情况、边界情况和异常情况

### 示例：添加新测试

```python
def test_your_feature():
    """测试你的新功能"""
    print("=== Test N: Your Feature ===")
    
    # 1. 准备
    event_bus = EventBus()
    adapter = MapServerAdapter(event_bus=event_bus)
    adapter.init_map()
    
    # 2. 执行
    result = adapter.your_method()
    
    # 3. 验证
    assert result is not None, "结果不应该为 None"
    print("✓ Test passed\n")
```

## 测试输出示例

```
============================================================
MapServerAdapter 单元测试
============================================================

=== Test 1: MapServerAdapter Initialization ===
✓ Test passed: Adapter initialized correctly

=== Test 2: Map Initialization ===
  📡 Event received: MAP_INITIALIZED
  📡 Event received: KNOWN_OBSTACLES_UPDATED (coords shape: (1234, 2))
  ✓ Map boundary: (0.00, 0.00) to (50.00, 50.00)
  ✓ Known obstacles: 1234 points
  ✓ Unknown obstacles: 10 points
✓ Test passed: Map initialized successfully

...

============================================================
✅ 所有测试通过！
============================================================
```

## 调试技巧

### 1. 打印详细信息
在测试中添加更多 print 语句来跟踪执行流程：

```python
print(f"  Debug: obstacle count = {len(adapter.known_obstacle_coordinates)}")
```

### 2. 使用断点
在关键位置添加断点：

```python
import pdb; pdb.set_trace()
```

### 3. 单独运行失败的测试
```python
# 只运行特定测试
test_lidar_scan_discovery()
```

### 4. 检查事件发布
添加事件监听器来查看所有事件：

```python
def debug_listener(*args, **kwargs):
    print(f"Event: args={args}, kwargs={kwargs}")

event_bus.subscribe('*', debug_listener)  # 注意：需要修改 EventBus 支持通配符
```

## 持续集成

将来可以在 CI/CD 中自动运行：

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/test_map_server.py -v
```

