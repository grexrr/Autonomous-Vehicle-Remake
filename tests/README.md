# 测试目录说明

本目录包含项目的所有测试文件，按照测试类型分类组织。

## 目录结构

```
tests/
├── unit/              # 单元测试 - 测试单个组件的功能
├── integration/       # 集成测试 - 测试多个组件协作
└── demo/              # 演示测试 - 算法演示和可视化测试
```

## 运行测试

### 方法 1：使用 pytest（推荐）

```bash
# 运行所有测试
pytest

# 运行特定目录
pytest tests/unit/
pytest tests/integration/

# 运行特定文件
pytest tests/unit/test_event_bus.py

# 详细输出
pytest -v

# 显示覆盖率
pytest --cov=api
```

### 方法 2：使用 unittest

```bash
# 运行所有测试
python -m unittest discover tests/

# 运行特定文件
python -m unittest tests.unit.test_event_bus
```

### 方法 3：直接运行（开发时）

```bash
# 运行单个测试文件
python -m tests.unit.test_event_bus
python -m tests.integration.test_routes
```

## 测试文件命名规范

- 测试文件必须以 `test_` 开头
- 测试函数必须以 `test_` 开头
- 测试类必须以 `Test` 开头

示例：
```python
# tests/unit/test_event_bus.py
def test_basic_event():  # ✓ 正确
    pass
```

## 添加新测试

1. 根据测试类型选择目录：
   - 单元测试 → `tests/unit/`
   - 集成测试 → `tests/integration/`
   - 演示测试 → `tests/demo/`

2. 创建测试文件，命名格式：`test_*.py`

3. 编写测试函数，命名格式：`test_*()`

4. 运行测试验证
