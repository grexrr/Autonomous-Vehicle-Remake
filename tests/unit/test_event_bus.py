import threading
import time
from api.adapters.event_bus import EventBus


def test_basic_event():
    print("=== Test 1: Basic Event ===")
    
    event_bus = EventBus()
    received_data = []

    def callback(data):
        received_data.append(data)
        print(f"Callback received data: {data}")
    
    # Subscribe
    event_bus.subscribe('test_event', callback)

    # Emit
    event_bus.emit('test_event', 'fuck my life')

    # Verify
    assert len(received_data) == 1, "Should receive 1 event"
    assert received_data[0] == 'fuck my life', "Data should match"
    print("✓ Test passed\n")

def test_multiple_subscribers():
    print("=== Test 2: Test Multiple Subscribers ===")
    
    event_bus = EventBus()
    results = []

    def callback1(data):
        results.append(f"callback1: {data}")
    
    def callback2(data):
        results.append(f"callback2: {data}")

    # subscribe 1 event
    event_bus.subscribe('update', callback1)
    event_bus.subscribe('update', callback2)

    # emit event
    event_bus.emit('update', 'fuck my life')

    # veryfy 2 callback functions are called
    assert len(results) == 2, "Should call 2 callbacks"
    print(f"Result: {results}")
    print("✓ Test passed\n")

def test_multiple_args():
    print("=== Test 3: Test Multiple Arguments ===")

    event_bus = EventBus()
    received_args = []

    def callback(timestamp, x, y):
        received_args.append((timestamp, x, y))
        print(f"Receiving: timestamp={timestamp}, x={x}, y={y}")
    
    event_bus.subscribe('state', callback)
    event_bus.emit('state', 1.0, 2.0, 3.0)

    assert received_args[0] == (1.0, 2.0, 3.0)
    print("✓ Test passed\n")

def test_kwargs():
    print("=== Test 4: Test Kwargs ===")

    event_bus = EventBus()
    received_kwargs = {}

    def callback(**kwargs):
        received_kwargs.update(kwargs)
        print(f"Receiving kwargs: {kwargs}")

    event_bus.subscribe('config', callback)
    event_bus.emit('config', host='localhost', port=5001)

    assert received_kwargs['host'] == 'localhost'
    assert received_kwargs['port'] == 5001
    print("✓ Test passed\n")

def test_thread_safety():
    print("=== Test 5: Thread Safety ===")
    
    event_bus = EventBus()
    results = []
    results_lock = threading.Lock()
    
    def callback(data):
        with results_lock:
            results.append(data)
        time.sleep(0.001) 
    
    subscribe_threads = []
    for _ in range(10):
        t = threading.Thread(target=lambda: event_bus.subscribe('test', callback))
        subscribe_threads.append(t)
        t.start()
    
    for t in subscribe_threads:
        t.join()
    
    emit_threads = []
    for _ in range(20):
        t = threading.Thread(target=lambda: event_bus.emit('test', f'data_{_}'))
        emit_threads.append(t)
        t.start()
    
    for t in emit_threads:
        t.join()
    
    expected_count = 20 * 10  # 20个事件 × 10个订阅者
    print(f'Expected: {expected_count}, Actual: {len(results)}')
    assert len(results) == expected_count, f"Expected {expected_count} results, got {len(results)}"
    print("✓ Test passed\n")

def test_unsubscribe():
    print("=== Test 6: Unsubscribe ===")
    
    event_bus = EventBus()
    results = []
    
    def callback(data):
        results.append(data)
    
    event_bus.subscribe('test', callback)
    event_bus.emit('test', 'data1')
    
    assert len(results) == 1
    
    event_bus.unsubscribe('test', callback)
    event_bus.emit('test', 'data2')
    
    # 取消订阅后不应该收到事件
    assert len(results) == 1, "Should not receive event after unsubscribe"
    print("✓ Test passed\n")


def test_clear():
    print("=== Test 7: Clear ===")
    
    event_bus = EventBus()
    results = []
    
    def callback(data):
        results.append(data)
    
    event_bus.subscribe('test1', callback)
    event_bus.subscribe('test2', callback)
    
    event_bus.clear()
    
    event_bus.emit('test1', 'data1')
    event_bus.emit('test2', 'data2')
    
    assert len(results) == 0, "Should not receive events after clear"
    print("✓ Test passed\n")


def test_no_subscribers():
    """测试没有订阅者时发出事件"""
    print("=== Test 8: No Subscribers ===")
    
    event_bus = EventBus()
    # 应该不会报错
    event_bus.emit('nonexistent_event', 'data')
    print("✓ Test passed (no error)\n")


if __name__ == '__main__':
    print("[Testing][EventBus]...\n")
    
    try:
        test_basic_event()
        test_multiple_subscribers()
        test_multiple_args()
        test_kwargs()
        test_thread_safety()
        test_unsubscribe()
        test_clear()
        test_no_subscribers()

        print("=" * 40)
        print("🎉[Testing][EventBus] All tests past!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
