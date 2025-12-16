from typing import Callable, Any, Dict, List

class EventBus:
    """
    Event Bus: Implements the publish-subscribe pattern

    Example usage:
        event_bus = EventBus()

        # Subscribe to event
        def handle_update(data):
            print(f"Received data: {data}")
        event_bus.subscribe('state_update', handle_update)

        # Emit event
        event_bus.emit('state_update', {'x': 1.0, 'y': 2.0})
    """

    def __init__(self):
        """
        初始化事件总线
        
        数据结构：
        _subscribers = {
            'event_name': [callback1, callback2, ...]
        }
        """
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type:str, callback: Callable) -> None:
        """
        Subscribe to an event.

        Args:
            event_type: Event name (string)
            callback: Callback function to be called when the event occurs

        Example:
            event_bus.subscribe('measured_state', my_function)
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    
    def emit(self, event_type: str, *args: Any, **kwargs:Any) -> None:
        """
        Emit an event and notify all subscribers.

        Args:
            event_type: Event name
            *args: Positional arguments to pass to the callback function
            **kwargs: Keyword arguments to pass to the callback function

        Example:
            event_bus.emit('measured_state', timestamp, car_object)
            event_bus.emit('state_update', timestamp=1.0, x=2.0)
        """
        
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(*args, **kwargs)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe (optional feature, may be needed in the future)
        
        Args:
            event_type: Event name
            callback: Callback function to be removed
        """
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
    
    def clear(self) -> None:
        """
        Clear all subscriptions
        """
        self._subscribers.clear()