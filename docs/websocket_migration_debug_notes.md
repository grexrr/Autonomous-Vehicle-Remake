# WebSocket Migration Debug Notes (Eventlet/Gevent)

This document summarizes the key debugging steps, findings, and conclusions
from local testing while preparing for EC2 deployment.

## Context

- Backend: Flask-SocketIO + Gunicorn in Docker.
- Frontend: Socket.IO client (React).
- Issue: With eventlet/gevent, WebSocket features broke (no rendered lines,
  second run fails, reconnect/invalid transport/session errors).

## Key Observations

- Forcing `transports: ['websocket']` on the client/server will *hard-fail*
  if the server does not provide websocket transport.
- In logs, errors like:
  - `Invalid transport`
  - `Invalid session`
  indicate the websocket transport is not properly available or session state
  is lost.

## What Was Verified

- Gunicorn worker class changed and logged correctly:
  - `eventlet` or `geventwebsocket` workers did boot as expected.
- The client was configured to prefer websocket.
- `Invalid transport` persisted even after rebuilds, pointing to server-side
  transport availability (not a frontend-only issue).

## Root Cause: Gevent/Eventlet vs Multiprocessing Pipes

The backend uses `multiprocessing.Pipe()` in adapters like:

- `api/adapters/global_planner_adapter.py`
- `api/adapters/local_planner_adapter.py`
- `api/adapters/process_adapter.py`

When gevent/eventlet monkey-patching is enabled, `os.read/os.write` can become
non-blocking. That breaks `multiprocessing.Connection.recv()` and `send()`,
causing:

- `BlockingIOError: [Errno 11] Resource temporarily unavailable`
- Child worker processes crash
- Gunicorn workers time out and restart
- WebSocket sessions break (`Invalid session/transport`)

This was reproduced even with:

- `monkey.patch_all(os=False, thread=False, subprocess=False)`
- Setting multiprocessing start method to `spawn`

### Why It Breaks

`multiprocessing.Connection.recv()` expects blocking IO:

```
BlockingIOError: [Errno 11] Resource temporarily unavailable
```

This error appeared inside:

```
api/adapters/global_planner_adapter.py
api/adapters/local_planner_adapter.py
```

Those adapters use `ProcessAdapter`, which wraps a `multiprocessing.Pipe()`
and starts child processes. Once monkey patching makes `os.read/os.write`
non-blocking, `pipe.recv()` fails and the worker process exits. That then
causes WebSocket session loss, `Invalid session`, and `Invalid transport`.

## Code Snippets That Matter

### Multiprocessing Pipe Usage (root of the issue)

`api/adapters/process_adapter.py`

```python
self._parent_pipe, self._child_pipe = mp.Pipe()
...
def _listen_loop(self) -> None:
    while self._running:
        if self._parent_pipe.poll(timeout=0.1):
            data = self._parent_pipe.recv()
            self._event_bus.emit(self._event_name, data)
```

`api/adapters/global_planner_adapter.py`

```python
def _worker_process(pipe: Connection, segment_collection_size: int) -> None:
    while True:
        match pipe.recv():
            case _ParentMsgType.CANCEL:
                continue
            case _ParentMsgType.PLAN, start, goal, obstacles:
                ...
```

### Gevent Worker With Monkey Patch (trigger)

`api/gevent_worker.py` (custom worker tried)

```python
from gevent import monkey
from geventwebsocket.gunicorn.workers import GeventWebSocketWorker

class PatchedGeventWebSocketWorker(GeventWebSocketWorker):
    def patch(self):
        monkey.patch_all(os=False, subprocess=False, thread=False)
```

Even with `os=False`, the worker still ended up with non-blocking behavior
because patching happens more than once, or because the worker itself patches
before our app code runs.

### Multiprocessing Start Method Attempt

`api/app.py`

```python
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
```

This was attempted to prevent inheriting monkey patches into child processes,
but it was not enough to avoid `BlockingIOError` under gevent workers.

## Attempts and Outcomes

1) **Eventlet worker + async_mode=eventlet**
   - WebSocket transport failed (`Invalid transport`).

2) **Gevent worker + async_mode=gevent**
   - WebSocket transport still failed.
   - When custom monkey patching avoided `os`, multiprocessing still threw
     `BlockingIOError` or workers timed out.

3) **Custom gevent worker (patched)**
   - Warned about signal patching.
   - HTTP health endpoint and StartSimulation could hang.
   - Gunicorn worker timed out.

## Error Logs Seen (Representative)

```
Invalid transport (further occurrences of this error will be logged with level INFO)
Invalid session _DMSQbOErYBRAEYDAAAA (further occurrences of this error will be logged with level INFO)
```

```
BlockingIOError: [Errno 11] Resource temporarily unavailable
```

```
[CRITICAL] WORKER TIMEOUT (pid:7)
[ERROR] Worker (pid:7) was sent SIGKILL! Perhaps out of memory?
```

```
MonkeyPatchWarning: Patching more than once will result in the union of all True parameters being patched
```

## Stable Path Found

**Use threading + simple-websocket**:

- Gunicorn worker class: `gthread`
- Flask-SocketIO async_mode: `threading`
- Dependency: `simple-websocket`
- Keep `transports=['websocket']` if you want to force websocket

### Example Config (Working)

`requirements.txt`

```txt
flask
flask-cors
flask-socketio
werkzeug>=2.3.0,<3.0.0
simple-websocket
gunicorn
python-dotenv
```

`api/app.py`

```python
async_mode = "threading"

socketio = SocketIO(
    app,
    cors_allowed_origins=app.config["SOCKETIO_CORS_ALLOWED_ORIGINS"],
    async_mode=async_mode,
    transports=["websocket"],
    ping_timeout=300,
    ping_interval=20,
)
```

`gunicorn_config.py`

```python
worker_class = "gthread"
thread = 4
```
