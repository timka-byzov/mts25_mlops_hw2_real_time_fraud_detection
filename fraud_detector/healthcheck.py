from pathlib import Path
import sys
import time

MAX_HEARTBEAT_AGE_SECONDS = 90
heartbeat = Path('/app/logs/heartbeat')
is_alive = heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < MAX_HEARTBEAT_AGE_SECONDS
sys.exit(0 if is_alive else 1)
