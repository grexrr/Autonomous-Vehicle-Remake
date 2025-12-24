import threading
import uuid
from typing import Dict, Optional

from api.session import UserSession

class SimulationManager:
    """
    Session Manager - manages multiple user sessions

    Responsibilities:
    1. Create a new session (returning session_id)
    2. Retrieve a session (by session_id)
    3. Delete a session (clean resources)
    4. Manage all sessions in a thread-safe way

    Design pattern: Singleton (only one manager needed for the whole application)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton Mode"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """
        Initialize the manager.

        Note: Because of the singleton pattern, __init__ may be called multiple times,
        so we need to check whether initialization has already occurred.
        """
        if hasattr(self, '_initialized'):
            return

        # Sessions Memo: {session_id: UserSession}
        self._sessions: Dict[str, UserSession] = {}

        self._sessions_lock = threading.RLock()
        self._initialized = True

    
    def create_session(self, initial_state: Optional[dict] = None, map_name: str = "map2") -> str:
        """
        Create a new session

        Args:
            initial_state: Optional initial state, e.g. {"x": 5.0, "y": 5.0, "yaw": 0.0}
             map_name: Map file name ("map" or "map2" or "map3"), default "map2"

        Returns:
            session_id: Newly created session ID
        """

        session_id = str(uuid.uuid4())
        user_session = UserSession(
            session_id=session_id, 
            initial_state=initial_state,
            map_name=map_name
        )

        with self._sessions_lock:
            self._sessions[session_id] = user_session
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get a session instance.
        
        Args:
            session_id: Session ID
        
        Returns:
            UserSession instance, or None if not found
        """
        with self._sessions_lock:
            return self._sessions.get(session_id)
        
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and clean up resources.
        
        Args:
            session_id: Session ID
        
        Returns:
            True if deletion was successful, False if the session does not exist.
        
        Important: You must call session.stop() to clean up resources!
        - Stop all timers
        - Stop all processes
        - Release memory
        """
        with self._sessions_lock:
            user_session = self._sessions.get(session_id)
            
            if user_session is None:
                return False    
            user_session.stop()
            del self._sessions[session_id]
            
            return True
    
    def list_sessions(self) -> list[str]:
        """
        List all active session IDs (for debugging purposes)
        
        Returns:
            List of all session IDs
        """
        with self._sessions_lock:
            return list(self._sessions.keys())
    
    def get_session_count(self) -> int:
        """
        Get the number of currently active sessions
        """
        with self._sessions_lock:
            return len(self._sessions)