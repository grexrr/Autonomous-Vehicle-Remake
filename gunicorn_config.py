import os
# Gunicorn Production Environment Configuration

# Number of worker processes
# If using WebSocket, it is recommended to use only 1 worker
# because multiple workers cannot share WebSocket connection state
workers = 1

# Worker class
worker_class = "gthread"
threads = 4

# Bind address and port
# 0.0.0.0 means listen on all network interfaces
# 5000 is the port (can be overridden via the PORT environment variable)
bind = "0.0.0.0:5000"

# Request timeout (seconds)
# Global planning algorithms may take longer, so this is set higher
timeout = 300

# Access log
accesslog= "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Logging level
loglevel = "info"

preload_app = False
