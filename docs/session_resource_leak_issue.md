# Session资源泄漏导致连接卡住问题

## 问题概述

**问题描述**: 前端多次进入/退出后，会卡在"Connecting..."状态，请求卡在创建session的步骤。重启前端无法修复，只有重启后端才能恢复。

**发现时间**: 2024年（待补充具体日期）

**严重程度**: 高 - 影响用户体验，需要重启后端才能恢复

**状态**: 已记录，待修复

---

## 问题现象

1. **首次进入**: 正常工作
2. **退出后再次进入**: 开始出现问题，偶尔会卡住
3. **多次进入/退出后**: 稳定复现，卡在"Connecting..."状态
4. **前端日志显示**:
   ```
   Start cleaning resources...
   Resources cleaned up
   Start creating session...
   请求 URL: http://localhost:5000/api/vehicle/session/create
   请求数据: Objectmap_name: "map"[[Prototype]]: Object
   ```
   然后卡住，不再继续

5. **重启前端**: 无法修复
6. **重启后端**: 可以恢复

---

## 根本原因分析

### 1. WebSocket断开时Session未被清理

**位置**: `api/websocket_handlers.py` - `handle_disconnect()`

**问题**:
- WebSocket断开时，后端只调用了 `session.unregister_websocket_push()`
- Session本身（包括所有进程、线程、定时器）**没有被停止或删除**
- Session仍然存在于 `SimulationManager._sessions` 字典中

**代码**:
```python
@socketio.on(WS_DISCONNECT)
def handle_disconnect(reason=None):
    # ...
    if session is not None:
        # 只清理了WebSocket引用，Session仍在运行
        session.unregister_websocket_push()
        # ❌ 没有调用 manager.delete_session(session_id)
        # ❌ 没有调用 session.stop()
```

### 2. 多次创建Session导致资源累积

**位置**: `api/simulation_manager.py` - `create_session()`

**问题**:
- 每次创建新session时，旧的session没有被清理
- 每个session包含：
  - 2个子进程（GlobalPlanner + LocalPlanner）
  - 多个线程（监听线程、定时器线程）
  - Pipe资源（进程间通信）
  - 内存占用

**累积效应**:
- 第1次: 1个session = 2个进程
- 第2次: 2个session = 4个进程
- 第3次: 3个session = 6个进程
- ...
- 多次后可能导致：
  - 系统进程数达到上限
  - Pipe资源耗尽
  - 内存泄漏
  - 新进程创建失败 → 卡住

### 3. ProcessAdapter.stop()可能清理不彻底

**位置**: `api/adapters/process_adapter.py` - `stop()`

**潜在问题**:
- 进程terminate/kill可能失败，导致僵尸进程
- Pipe关闭异常被吞掉（try-except pass），可能未完全关闭
- 重新创建pipe时，如果旧的未完全关闭，可能冲突

**代码**:
```python
def stop(self) -> None:
    # ...
    try:
        if not self._parent_pipe.closed:
            self._parent_pipe.close()
    except:
        pass  # ❌ 异常被吞掉，可能未完全关闭
```

### 4. 前端缺少connect_error监听

**位置**: `vite-project/src/hooks/useAutonomousVehicle.js`

**问题**:
- 当后端拒绝连接时（如session创建失败），Socket.IO会触发 `connect_error` 事件
- 前端只监听了 `WS_ERROR`，没有监听 `connect_error`
- 导致UI卡在"Connecting..."状态，无法显示错误信息

**代码**:
```javascript
// ✅ 有监听
socket.on(WS_ERROR, (data) => {
  console.error("WebSocket error:", data.message);
  setConnectionStatus(CONNECTION_STATUS.DISCONNECTED);
});

// ❌ 缺少监听
// socket.on("connect_error", (err) => {
//   console.error("connect_error:", err.message);
//   setConnectionStatus(CONNECTION_STATUS.DISCONNECTED);
// });
```

### 5. Session创建可能阻塞

**位置**: `api/session.py` - `UserSession.__init__()` → `_initialize()`

**问题**:
- `_initialize()` 同步调用，会启动多个进程
- 如果系统资源不足（进程数、文件描述符等），可能阻塞：
  - `ProcessAdapter.start()` 创建进程可能失败
  - `mp.Pipe()` 创建可能失败
  - 进程启动可能卡住

**代码**:
```python
def __init__(self, session_id: str, ...):
    # ...
    self._setup_event_handlers()
    self._initialize()  # ❌ 同步调用，可能阻塞

def _initialize(self) -> None:
    self.map_server.init_map(self._map_name)
    self.car_simulation.start()
    self.global_planner.start()  # 启动进程，可能失败
    self.local_planner.start()   # 启动进程，可能失败
```

---

## 影响范围

1. **用户体验**: 多次使用后系统不可用，需要重启后端
2. **资源消耗**: 进程、线程、内存持续累积
3. **系统稳定性**: 可能导致系统资源耗尽

---

## 相关代码位置

### 后端
- `api/websocket_handlers.py`: `handle_disconnect()` - WebSocket断开处理
- `api/simulation_manager.py`: `create_session()` - Session创建
- `api/session.py`: `UserSession.__init__()`, `_initialize()`, `stop()` - Session生命周期
- `api/adapters/process_adapter.py`: `stop()` - 进程清理

### 前端
- `vite-project/src/hooks/useAutonomousVehicle.js`: `createSession()`, WebSocket事件监听

---

## 建议的修复方案

### 方案1: WebSocket断开时自动清理Session（推荐）

**优点**: 简单直接，符合用户预期

**实现**:
```python
@socketio.on(WS_DISCONNECT)
def handle_disconnect(reason=None):
    # ...
    if session_id:
        manager = SimulationManager()
        # 延迟删除，给重连留出时间窗口
        # 或者立即删除，前端需要重新创建session
        manager.delete_session(session_id)
```

**考虑**: 如果用户只是短暂断开（网络波动），立即删除可能不合适。可以：
- 添加超时机制：断开后N秒内未重连才删除
- 或者：前端断开时主动调用DELETE接口

### 方案2: 创建新Session前清理旧Session

**优点**: 确保不会累积

**实现**:
```python
def create_session(self, ...):
    # 清理所有未使用的session（没有WebSocket连接的）
    self._cleanup_orphaned_sessions()
    
    session_id = str(uuid.uuid4())
    # ...
```

### 方案3: 改进ProcessAdapter.stop()的错误处理

**实现**:
```python
def stop(self) -> None:
    # ...
    try:
        if not self._parent_pipe.closed:
            self._parent_pipe.close()
    except Exception as e:
        print(f"[ProcessAdapter] Error closing parent pipe: {e}")  # 记录错误
        # 尝试强制关闭
        try:
            self._parent_pipe.close()
        except:
            pass
```

### 方案4: 前端添加connect_error监听

**实现**:
```javascript
socket.on("connect_error", (err) => {
  console.error("connect_error:", err.message);
  setConnectionStatus(CONNECTION_STATUS.DISCONNECTED);
  // 可选：显示错误提示给用户
});
```

### 方案5: 添加资源监控和日志

**实现**:
- 记录每个session的创建/删除时间
- 监控进程数、线程数
- 添加健康检查接口，显示当前session数量

---

## 测试建议

1. **压力测试**: 连续创建/删除10个session，检查资源是否正常释放
2. **长时间运行**: 运行24小时，检查是否有资源泄漏
3. **异常场景**: 模拟进程创建失败、Pipe创建失败等情况

---

## 相关Issue/PR

- 待补充

---

## 备注

- 此问题在开发环境中稳定复现
- 生产环境可能因为资源限制更容易触发
- 建议优先修复方案1和方案4（最简单有效）

