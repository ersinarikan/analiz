"""
Training tasks - Eğitim görevleri için Celery tasks
"""
import logging

logger = logging.getLogger(__name__)

def start_training_task(model_type, training_params):
    """
    Eğitim görevini başlatır - SSE sistemi kullanır
    """
    logger.info(f"Eğitim görevi başlatıldı: {model_type}")
    # SSE sistemi tarafından işlenir
    return {"status": "started", "message": "Eğitim SSE sistemi tarafından yönetiliyor"}

def start_evaluation_task(model_version):
    """
    Model değerlendirme görevini başlatır
    """
    logger.info(f"Model değerlendirme başlatıldı: {model_version}")
    # SSE sistemi tarafından işlenir
    return {"status": "started", "message": "Değerlendirme SSE sistemi tarafından yönetiliyor"} 