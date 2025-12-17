# Global Planner Adapter 学习笔记：进程、线程与事件机制

## 📚 基础概念

### 1. 进程 (Process) vs 线程 (Thread)

**进程 (Process)**
- 进程是**独立的执行单元**，拥有自己的内存空间
- 不同进程之间的数据**不能直接共享**
- 进程间通信需要特殊机制（如 Pipe、Queue）
- 优点：隔离性好，一个进程崩溃不会影响其他进程
- 缺点：创建和通信开销较大

**线程 (Thread)**
- 线程是进程内的**执行流**，共享进程的内存空间
- 同一进程内的线程可以**直接访问共享数据**
- 优点：创建和切换开销小，通信简单
- 缺点：需要锁机制防止数据竞争

**为什么这里要用进程？**
- 路径规划算法（hybrid_a_star）是**CPU密集型任务**
- 放在独立进程中可以：
  - 避免阻塞主程序
  - 利用多核CPU
  - 设置高优先级提升性能

---

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    主进程 (Main Process)                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         GlobalPlannerAdapter                         │   │
│  │  - plan() 方法：发送规划请求                            │   │
│  │  - _handle_worker_message()：处理子进程消息             │   │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          │ 1. 创建并启动                      │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         ProcessAdapter                                │   │
│  │  - 管理子进程                                          │   │
│  │  - 管理监听线程                                        │   │
│  │  - 通过 EventBus 发布消息                             │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                    │                                │
│         │ 2. 创建            │ 3. 启动监听线程                │
│         ▼                    ▼                                │
│  ┌──────────────┐    ┌──────────────────────────────┐       │
│  │  子进程       │    │   监听线程                    │       │
│  │ (Worker)     │◄───┤  (_listen_loop)              │       │
│  │              │    │  - 持续监听 pipe              │       │
│  │ _worker_     │    │  - 收到消息后发布事件         │       │
│  │ process()    │    └──────────────────────────────┘       │
│  └──────────────┘                                            │
│         │                                                    │
│         │ 通过 Pipe 通信                                      │
│         └───────────────────────────────────────────────────┘
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         EventBus (事件总线)                            │   │
│  │  - 维护订阅者列表                                       │   │
│  │  - 发布事件时调用所有订阅者的回调函数                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 完整数据流详解

### 阶段 1：初始化 (Initialization)

```python
# 在 GlobalPlannerAdapter.__init__() 中

# 1. 创建 ProcessAdapter
self._process_adapter = ProcessAdapter(
    event_bus=event_bus,                    # 共享的 EventBus 实例
    event_name='_global_planner_worker_message',  # 内部事件名
    target=_worker_process,                 # 子进程要运行的函数
    args=(segment_collection_size,)         # 传给子进程的参数
)
```

**发生了什么？**
- `ProcessAdapter` 创建了一个 **Pipe**（管道）用于进程间通信
  - `_parent_pipe`：主进程端
  - `_child_pipe`：子进程端
- 创建了子进程对象（但还没启动）

```python
# 2. 订阅内部事件
self._event_bus.subscribe(
    '_global_planner_worker_message', 
    self._handle_worker_message
)
```

**发生了什么？**
- `GlobalPlannerAdapter` 告诉 EventBus：
  - "当有人发布 `'_global_planner_worker_message'` 事件时，请调用我的 `_handle_worker_message` 方法"
- EventBus 内部维护一个字典：
  ```python
  _subscribers = {
      '_global_planner_worker_message': [self._handle_worker_message]
  }
  ```

```python
# 3. 启动子进程和监听线程
self._process_adapter.start()
```

**发生了什么？**
- 启动子进程：`_worker_process` 函数开始在另一个进程中运行
- 启动监听线程：`_listen_loop` 开始在主进程中运行（后台线程）

---

### 阶段 2：发送规划请求 (Sending Planning Request)

```python
# 用户调用 GlobalPlannerAdapter.plan()
adapter.plan(start_state, goal_state, obstacles)
```

**数据流：**

```
GlobalPlannerAdapter.plan()
    │
    │ 1. 准备数据
    │    start = [x, y, yaw]
    │    goal = [x, y, yaw]
    │
    ▼
ProcessAdapter.send((_ParentMsgType.PLAN, start, goal, obstacles))
    │
    │ 2. 通过 Pipe 发送到子进程
    │    _parent_pipe.send(...)
    │
    ▼
子进程中的 _worker_process()
    │
    │ 3. 接收消息
    │    pipe.recv()  # 这里 pipe 就是 _child_pipe
    │
    ▼
开始执行 hybrid_a_star 算法
```

**关键点：**
- `send()` 是**同步的**：数据立即写入 Pipe
- 子进程通过 `pipe.recv()` **阻塞等待**消息（在 while True 循环中）

---

### 阶段 3：子进程计算并发送结果 (Worker Computing & Sending)

```python
# 在子进程 _worker_process() 中

# 执行路径规划算法
traj = hybrid_a_star(start, goal, obstacles, callback)

# 发送最终结果
pipe.send((_WorkerMsgType.TRAJECTORY, traj))
```

**数据流：**

```
子进程 _worker_process()
    │
    │ 1. 执行计算（CPU 密集型）
    │    hybrid_a_star(...)
    │
    │ 2. 发送结果
    │    pipe.send((TRAJECTORY, traj))
    │    这里的 pipe 是 _child_pipe
    │
    ▼
Pipe (_child_pipe → _parent_pipe)
    │
    │ 3. 数据通过操作系统内核传递
    │
    ▼
主进程的 _parent_pipe
```

**关键点：**
- 子进程在**独立的内存空间**中运行
- 通过 Pipe 发送的数据会被**序列化**（pickle），然后传递到主进程

---

### 阶段 4：监听线程接收消息 (Listener Thread Receiving)

```python
# 在 ProcessAdapter._listen_loop() 中（运行在监听线程）

while self._running:
    if self._parent_pipe.poll(timeout=0.1):  # 检查是否有数据
        data = self._parent_pipe.recv()       # 接收数据
        self._event_bus.emit(self._event_name, data)  # 发布事件
```

**数据流：**

```
监听线程 _listen_loop()
    │
    │ 1. 持续轮询 Pipe（每 0.1 秒检查一次）
    │    _parent_pipe.poll(timeout=0.1)
    │
    │ 2. 收到数据
    │    data = _parent_pipe.recv()
    │    data = (_WorkerMsgType.TRAJECTORY, traj)
    │
    │ 3. 发布事件到 EventBus
    │    event_bus.emit('_global_planner_worker_message', data)
    │
    ▼
EventBus.emit()
    │
    │ 4. EventBus 查找所有订阅者
    │    callbacks = _subscribers['_global_planner_worker_message']
    │
    │ 5. 调用每个订阅者的回调函数
    │    callback(data)  # 即 _handle_worker_message(data)
    │
    ▼
GlobalPlannerAdapter._handle_worker_message(data)
```

**关键点：**
- 监听线程是**独立的执行流**，不会阻塞主程序
- `poll(timeout=0.1)` 是**非阻塞**的：如果没有数据，等待 0.1 秒后继续循环
- EventBus 的 `emit()` 是**同步的**：会立即调用所有订阅者的回调函数

---

### 阶段 5：处理消息并发布最终事件 (Handling Message & Publishing Final Event)

```python
# 在 GlobalPlannerAdapter._handle_worker_message() 中

def _handle_worker_message(self, data):
    match data:
        case _WorkerMsgType.TRAJECTORY, trajectory:
            # 发布最终轨迹事件
            self._event_bus.emit(GLOBAL_PLANNER_TRAJECTORY, trajectory)
            
            if trajectory is not None:
                # 发布完成事件
                self._event_bus.emit(GLOBAL_PLANNER_FINISHED)
```

**数据流：**

```
_handle_worker_message(data)
    │
    │ 1. 解包消息
    │    data = (_WorkerMsgType.TRAJECTORY, traj)
    │
    │ 2. 发布公开事件
    │    event_bus.emit('global_planner_trajectory', traj)
    │    event_bus.emit('global_planner_finished')
    │
    ▼
其他订阅者（如 UI 组件、其他适配器）
    │
    │ 3. 这些组件已经订阅了公开事件
    │    event_bus.subscribe('global_planner_trajectory', ...)
    │
    ▼
更新 UI 或执行其他操作
```

**关键点：**
- `_handle_worker_message` 是**事件处理器**：它接收内部消息，然后发布公开事件
- 这样设计的好处：
  - **解耦**：GlobalPlannerAdapter 不需要知道谁在使用这些事件
  - **灵活性**：多个组件可以订阅同一个事件

---

## 🎯 关键问题解答

### Q1: 谁发布事件？谁监听事件？

**发布者 (Publisher)：**
1. **ProcessAdapter** 发布 `'_global_planner_worker_message'` 事件
   - 发布时机：监听线程收到子进程消息时
   - 位置：`ProcessAdapter._listen_loop()` 第 133 行

2. **GlobalPlannerAdapter** 发布 `'global_planner_trajectory'` 等公开事件
   - 发布时机：处理完子进程消息后
   - 位置：`GlobalPlannerAdapter._handle_worker_message()` 第 164-170 行

**监听者 (Subscriber)：**
1. **GlobalPlannerAdapter** 监听 `'_global_planner_worker_message'`
   - 订阅时机：初始化时（第 119 行）
   - 响应方法：`_handle_worker_message()`

2. **其他组件**（如 UI、其他适配器）监听 `'global_planner_trajectory'` 等
   - 订阅时机：各自初始化时
   - 响应方法：各自定义的回调函数

### Q2: 如何监听和响应？

**EventBus 的工作原理：**

```python
# 订阅（注册监听）
event_bus.subscribe('event_name', callback_function)
# EventBus 内部：
# _subscribers['event_name'].append(callback_function)

# 发布（触发事件）
event_bus.emit('event_name', data)
# EventBus 内部：
# for callback in _subscribers['event_name']:
#     callback(data)  # 调用所有订阅者的回调函数
```

**监听流程：**
1. 组件调用 `subscribe()` 注册回调函数
2. EventBus 将回调函数存储在字典中
3. 当有人 `emit()` 事件时，EventBus 查找所有订阅者
4. EventBus **同步调用**所有订阅者的回调函数

### Q3: 为什么需要监听线程？

**问题：** 为什么不能直接在主线程中接收 Pipe 消息？

**答案：**
- `pipe.recv()` 是**阻塞的**：如果没有数据，会一直等待
- 如果在主线程中阻塞，整个程序会卡住
- 解决方案：在**独立的线程**中监听，主线程继续运行

**监听线程的作用：**
- 持续检查 Pipe 是否有数据（非阻塞轮询）
- 收到数据后立即发布事件
- 不阻塞主程序的其他操作

### Q4: 进程间通信是如何工作的？

**Pipe (管道) 机制：**

```
主进程                   子进程
  │                        │
  │                        │
_parent_pipe          _child_pipe
  │                        │
  │                        │
  └───────── Pipe ─────────┘
    (操作系统内核管理)
```

- Pipe 是操作系统提供的**进程间通信机制**
- 数据会被**序列化**（pickle）后传递
- 发送和接收是**同步的**：发送方会等待接收方接收（或缓冲区满）

---

## 📝 代码位置索引

### ProcessAdapter
- **创建 Pipe**：第 70 行
- **启动子进程和线程**：第 81-101 行
- **发送消息**：第 103-105 行
- **监听循环**：第 124-138 行

### GlobalPlannerAdapter
- **初始化 ProcessAdapter**：第 112-117 行
- **订阅内部事件**：第 119 行
- **发送规划请求**：第 125-146 行
- **处理子进程消息**：第 152-170 行

### EventBus
- **订阅**：第 33-47 行
- **发布**：第 50-74 行

---

## 💡 设计模式总结

1. **进程间通信模式**：使用 Pipe 在独立进程中执行 CPU 密集型任务
2. **发布-订阅模式**：使用 EventBus 实现组件间解耦通信
3. **适配器模式**：GlobalPlannerAdapter 适配原有的 GlobalPlannerNode 接口

---

## 🔍 调试技巧

**如何追踪数据流？**

1. 在关键位置添加打印：
   ```python
   # ProcessAdapter._listen_loop()
   print(f"[Listener] Received: {data}")
   
   # GlobalPlannerAdapter._handle_worker_message()
   print(f"[Adapter] Handling: {data}")
   ```

2. 检查进程和线程状态：
   ```python
   print(f"Process alive: {adapter.is_alive()}")
   ```

3. 验证事件订阅：
   ```python
   # 在 EventBus.emit() 中添加日志
   print(f"[EventBus] Emitting '{event_type}' to {len(callbacks)} subscribers")
   ```
