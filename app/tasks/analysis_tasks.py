"""
Analysis tasks - Analiz görevleri için task definitions
"""
import logging

logger = logging.getLogger(__name__)

def run_analysis_task(file_id, file_path, analysis_params):
    """
    Analiz görevini çalıştırır - HTTP API kullanır
    """
    logger.info(f"Analiz görevi başlatıldı: {file_id}")
    # HTTP API tarafından işlenir
    return {"status": "started", "message": "Analiz HTTP API tarafından yönetiliyor"}

def run_batch_analysis_task(file_ids, analysis_params):
    """
    Toplu analiz görevini çalıştırır
    """
    logger.info(f"Toplu analiz başlatıldı: {len(file_ids)} dosya")
    # HTTP API tarafından işlenir
    return {"status": "started", "message": "Toplu analiz HTTP API tarafından yönetiliyor"} 