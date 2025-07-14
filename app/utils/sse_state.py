import threading
import time
from typing import Dict, List, Any
from datetime import datetime


class SSETrainingState:
    """Thread-safe state manager for training progress via Server-Sent Events"""

    def __init__(self):
        self._sessions = {}  # session_id -> {events: [], created_at: datetime}
        self._lock = threading.Lock()
        
        # Cleanup thread for old sessions
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_sessions, daemon=True
        )
        self._cleanup_thread.start()

    def create_session(self, session_id: str, training_type: str = 'content'):
        """Create a new training session"""
        with self._lock:
            self._sessions[session_id] = {
                'events': [{
                    'type': 'connected',
                    'session_id': session_id,
                    'training_type': training_type,
                    'timestamp': datetime.now().isoformat()
                }],
                'created_at': datetime.now(),
                'training_type': training_type
            }

    def add_event(self, session_id: str, event_data: Dict[str, Any]):
        """Add an event to the session"""
        with self._lock:
            if session_id in self._sessions:
                event_data['timestamp'] = datetime.now().isoformat()
                self._sessions[session_id]['events'].append(event_data)

    def get_events(self, session_id: str, since_index: int = 0):
        """Get events for a session since the given index"""
        with self._lock:
            if session_id not in self._sessions:
                return []
            
            events = self._sessions[session_id]['events']
            return events[since_index:]

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        with self._lock:
            return session_id in self._sessions

    def cleanup_session(self, session_id: str):
        """Remove a session"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def get_session_info(self, session_id: str):
        """Get session information"""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            return None

    def _cleanup_old_sessions(self):
        """Background thread to cleanup old sessions"""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                cutoff_time = datetime.now().timestamp() - 3600  # 1 hour ago
                
                with self._lock:
                    sessions_to_remove = [
                        sid for sid, data in self._sessions.items()
                        if data['created_at'].timestamp() < cutoff_time
                    ]
                    
                    for sid in sessions_to_remove:
                        del self._sessions[sid]
                    
                    if sessions_to_remove:
                        print(f"[SSE] Cleaned up {len(sessions_to_remove)} "
                              f"old training sessions")
            except Exception as e:
                print(f"[SSE] Cleanup error: {e}")


# Global instance
sse_training_state = SSETrainingState()


def emit_sse_event(session_id: str, event_type: str, data: Dict[str, Any]):
    """Emit an SSE event for the given session"""
    event_data = {
        'type': event_type,
        'session_id': session_id,
        **data
    }
    sse_training_state.add_event(session_id, event_data)


def emit_training_started(session_id: str, training_type: str = 'content',
                          **kwargs):
    """Emit training started event"""
    emit_sse_event(session_id, 'training_started', {
        'training_type': training_type,
        **kwargs
    })


def emit_training_progress(session_id: str, current_epoch: int,
                           total_epochs: int, current_loss: float,
                           current_mae: float = None,
                           current_r2: float = None, **kwargs):
    """Emit training progress event"""
    progress_data = {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'current_loss': current_loss,
        **kwargs
    }
    
    if current_mae is not None:
        progress_data['current_mae'] = current_mae
    if current_r2 is not None:
        progress_data['current_r2'] = current_r2
        
    emit_sse_event(session_id, 'training_progress', progress_data)


def emit_training_completed(session_id: str, model_version: str, **kwargs):
    """Emit training completed event"""
    emit_sse_event(session_id, 'training_completed', {
        'model_version': model_version,
        **kwargs
    })


def emit_training_error(session_id: str, error_message: str, **kwargs):
    """Emit training error event"""
    emit_sse_event(session_id, 'training_error', {
        'error': error_message,
        **kwargs
    }) 