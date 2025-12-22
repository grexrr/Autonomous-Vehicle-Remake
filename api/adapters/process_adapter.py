import multiprocessing as mp
import threading
from multiprocessing.connection import Connection
from typing import Any, Optional, Protocol
import sys
from pathlib import Path

# PROJECT ROOT
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AutonomousVehicle.utils.set_high_priority import set_high_priority
from .event_bus import EventBus

class CallableWithConnection(Protocol):
    """Protocol for worker functions that accept a Connection as first argument"""
    def __call__(self, pipe: Connection, *args: Any, **kwargs: Any) -> None: ...


class ProcessAdapter:
    """
    Process communication adapter using EventBus instead of Qt Signals
    
    This class:
    1. Creates a child process to run CPU-intensive tasks
    2. Starts a listener thread to receive messages from child process
    3. Publishes received messages via EventBus
    
    Usage:
        def worker_func(pipe: Connection, arg1, arg2):
            # Heavy computation in separate process
            result = compute(arg1, arg2)
            pipe.send(result)
        
        adapter = ProcessAdapter(
            event_bus=event_bus,
            event_name='result_received',
            target=worker_func,
            args=(value1, value2)
        )
        adapter.start()
        adapter.send(data)  # Send message to child process
    """

    def __init__(
        self,
        event_bus: EventBus,
        event_name: str,
        target: CallableWithConnection, # working function
        args: tuple = (),
        kwargs: Optional[dict] = None
    ) -> None:
        """
        Initialize process adapter
        
        Args:
            event_bus: EventBus instance for publishing messages
            event_name: Event name to publish when receiving messages
            target: Worker function to run in child process (first arg must be Connection)
            args: Arguments to pass to worker function (after pipe)
            kwargs: Keyword arguments to pass to worker function
        """
        self._event_bus = event_bus
        self._event_name = event_name
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        
        # Store target and args for potential restart
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        
        # create pipe for inter-process comms
        self._parent_pipe, self._child_pipe = mp.Pipe()

        # create child process
        self._child_process = mp.Process(
            target=target,
            args=(self._child_pipe, *args),
            kwargs=self._kwargs,
            daemon=True
        )
    
    def start(self) -> None:
        """Start the child process and listener thread"""
        if self._running:
            return
        self._running = True

        # 1. start
        self._child_process.start()
        
        # 2. set high priority for working process
        try:
            set_high_priority(self._child_process.pid)
        except Exception as e:
            print(f"[ProcessAdapter] Could not set high priority: {e}")
        
        # 3. start listener thread
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self._listener_thread.start()
    
    def send(self, obj: Any) -> None:
        if self._running:
            self._parent_pipe.send(obj)
    
    def stop(self) -> None:
        """Stop the child process and listener thread"""
        if not self._running:
            return
        
        self._running = False

        # Terminate and wait for process
        if hasattr(self._child_process, 'is_alive') and self._child_process.is_alive():
            self._child_process.terminate()
            self._child_process.join(timeout=1.0)
            if self._child_process.is_alive():
                self._child_process.kill()
                self._child_process.join(timeout=1.0)
        
        # Wait for listener thread
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)
        
        # Close pipes
        try:
            if not self._parent_pipe.closed:
                self._parent_pipe.close()
        except:
            pass
        try:
            if not self._child_pipe.closed:
                self._child_pipe.close()
        except:
            pass
        
        # Recreate pipes and process for potential restart
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._child_process = mp.Process(
            target=self._target,
            args=(self._child_pipe, *self._args),
            kwargs=self._kwargs,
            daemon=True
        )

    def _listen_loop(self) -> None:
        """
        Continuously listen for messages from child process
        (Runs in separate thread)
        """
        while self._running:
            try:
                if self._parent_pipe.poll(timeout=0.1):
                    data = self._parent_pipe.recv()
                    self._event_bus.emit(self._event_name, data)
            except EOFError:
                break
            except Exception as e:
                print(f"[ProcessAdapter] Error in listener thread: {e}")
                break

    def is_alive(self) -> bool:
        return self._child_process.is_alive()