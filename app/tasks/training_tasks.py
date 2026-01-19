"""
Training tasks - Eğitim görevleri için Celery tasks
"""
import logging

logger = logging.getLogger(__name__)

def start_training_task(training_data, model_type='content'):
    """
    Eğitim görevini başlatır - WebSocket sistemi kullanır
    """
    try:
        # WebSocket sistemi tarafından işlenir
        return {"status": "started", "message": "Eğitim WebSocket sistemi tarafından yönetiliyor"}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def start_evaluation_task(model_id):
    """
    Model değerlendirme görevini başlatır
    """
    try:
        # WebSocket sistemi tarafından işlenir
        return {"status": "started", "message": "Değerlendirme WebSocket sistemi tarafından yönetiliyor"}
        
    except Exception as e:
        return {"status": "error", "error": str(e)} 