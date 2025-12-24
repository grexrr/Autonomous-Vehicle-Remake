"""
ProcessAdapter 单元测试
测试进程通信适配器的各项功能
"""

import time
import threading
from multiprocessing.connection import Connection

from api.adapters.event_bus import EventBus
from api.adapters.process_adapter import ProcessAdapter


# ========== Worker Functions (must be at module level for pickle) ==========

def dummy_worker(pipe: Connection, arg1: int = 0, arg2: str = ""):
    """简单的测试工作函数"""
    pass


def simple_worker(pipe: Connection):
    """简单的工作函数，等待消息"""
    while True:
        msg = pipe.recv()
        if msg == 'stop':
            break


def echo_worker(pipe: Connection):
    """回显工作函数，将收到的消息发回"""
    while True:
        msg = pipe.recv()
        if msg == 'stop':
            break
        pipe.send(f"echo: {msg}")


def counting_worker(pipe: Connection):
    """计数工作函数，收到消息就发送计数"""
    count = 0
    while True:
        msg = pipe.recv()
        if msg == 'stop':
            break
        count += 1
        pipe.send(count)


def worker_with_args(pipe: Connection, multiplier: int, prefix: str):
    """带参数的工作函数"""
    while True:
        msg = pipe.recv()
        if msg == 'stop':
            break
        result = f"{prefix}: {msg * multiplier}"
        pipe.send(result)


def test_initialization():
    """测试初始化"""
    print("=== Test 1: Initialization ===")
    
    event_bus = EventBus()
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='test_event',
        target=dummy_worker,
        args=(10, 'test')
    )
    
    assert adapter._event_bus == event_bus
    assert adapter._event_name == 'test_event'
    assert not adapter._running
    print("✓ Test passed\n")


def test_start_stop():
    """测试启动和停止"""
    print("=== Test 2: Start/Stop ===")
    
    event_bus = EventBus()
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='test_event',
        target=simple_worker
    )
    
    # 启动
    adapter.start()
    
    # 等待工作进程启动
    time.sleep(0.2)
    assert adapter.is_alive(), "进程应该正在运行"
    
    # 停止
    adapter.send('stop')
    time.sleep(0.1)
    adapter.stop()
    
    # 等待进程完全停止
    time.sleep(0.2)
    assert not adapter.is_alive(), "进程应该已停止"
    
    print("✓ Test passed\n")


def test_send_receive():
    """测试发送和接收消息"""
    print("=== Test 3: Send/Receive ===")
    
    event_bus = EventBus()
    received_messages = []
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='echo_event',
        target=echo_worker
    )
    
    def on_message(data):
        received_messages.append(data)
        print(f"  收到消息: {data}")
    
    # 订阅事件
    event_bus.subscribe('echo_event', on_message)
    
    # 启动
    adapter.start()
    time.sleep(0.1)
    
    # 发送消息
    adapter.send('hello')
    time.sleep(0.2)  # 等待消息处理
    
    # 验证收到回显
    assert len(received_messages) > 0, "应该收到消息"
    assert any('echo: hello' in str(msg) for msg in received_messages), "应该收到回显消息"
    
    # 停止
    adapter.send('stop')
    time.sleep(0.1)
    adapter.stop()
    
    print("✓ Test passed\n")


def test_multiple_messages():
    """测试多条消息"""
    print("=== Test 4: Multiple Messages ===")
    
    event_bus = EventBus()
    received_count = [0]  # 使用列表以便在闭包中修改
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='count_event',
        target=counting_worker
    )
    
    def on_message(data):
        received_count[0] += 1
        print(f"  收到计数: {data}")
    
    event_bus.subscribe('count_event', on_message)
    
    adapter.start()
    time.sleep(0.1)
    
    # 发送多条消息
    for i in range(5):
        adapter.send(f'message_{i}')
        time.sleep(0.1)
    
    # 验证收到多条消息
    assert received_count[0] >= 5, f"应该收到至少5条消息，实际收到{received_count[0]}"
    
    adapter.send('stop')
    time.sleep(0.1)
    adapter.stop()
    
    print("✓ Test passed\n")


def test_worker_with_args():
    """测试带参数的工作函数"""
    print("=== Test 5: Worker With Args ===")
    
    event_bus = EventBus()
    received_data = []
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='result_event',
        target=worker_with_args,
        args=(3, 'RESULT')
    )
    
    def on_result(data):
        received_data.append(data)
        print(f"  收到结果: {data}")
    
    event_bus.subscribe('result_event', on_result)
    
    adapter.start()
    
    # 等待进程启动并验证进程是否存活
    time.sleep(0.2)  # 增加等待时间
    assert adapter.is_alive(), "进程应该已启动"
    
    adapter.send(5)
    time.sleep(0.3)  # 增加等待时间，确保消息被处理
    
    # 验证参数被正确传递
    assert len(received_data) > 0, f"应该收到结果，但 received_data={received_data}"
    assert any('RESULT' in str(data) and '15' in str(data) for data in received_data), \
        f"应该看到前缀和计算结果，但 received_data={received_data}"
    
    adapter.send('stop')
    time.sleep(0.1)
    adapter.stop()
    
    print("✓ Test passed\n")


def test_stop_before_start():
    """测试在启动前停止"""
    print("=== Test 6: Stop Before Start ===")
    
    event_bus = EventBus()
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='test_event',
        target=dummy_worker
    )
    
    # 在启动前停止应该不会报错
    adapter.stop()
    
    print("✓ Test passed\n")


def test_multiple_start_calls():
    """测试多次调用 start()"""
    print("=== Test 7: Multiple Start Calls ===")
    
    event_bus = EventBus()
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='test_event',
        target=simple_worker
    )
    
    # 多次调用 start
    adapter.start()
    time.sleep(0.1)
    adapter.start()  # 第二次调用应该被忽略
    time.sleep(0.1)
    
    assert adapter.is_alive(), "进程应该正在运行"
    
    adapter.send('stop')
    time.sleep(0.1)
    adapter.stop()
    
    print("✓ Test passed\n")


def test_send_when_not_running():
    """测试在未运行时发送消息"""
    print("=== Test 8: Send When Not Running ===")
    
    event_bus = EventBus()
    
    adapter = ProcessAdapter(
        event_bus=event_bus,
        event_name='test_event',
        target=dummy_worker
    )
    
    # 未启动时发送消息应该不会报错（但可能不会发送）
    adapter.send('test')
    
    print("✓ Test passed\n")


if __name__ == '__main__':
    print("[Testing][ProcessAdapter]...\n")
    
    try:
        test_initialization()
        test_start_stop()
        test_send_receive()
        test_multiple_messages()
        test_worker_with_args()
        test_stop_before_start()
        test_multiple_start_calls()
        test_send_when_not_running()
        
        print("=" * 40)
        print("🎉[Testing][ProcessAdapter] All tests passed!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

