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


if __name__ == '__main__':
    print("[Testing][Event Bus]...\n")
    
    try:
        test_basic_event()
        test_multiple_subscribers()
        print("=" * 40)
        print("🎉[Testing][Event Bus] All tests past!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
