# Model State File
# Bu dosya model aktivasyonlarında güncellenir ve Flask debug mode tarafından izlenir
# Otomatik restart için config.py tarafından import edilir

import threading
import time
from datetime import datetime
import weakref
import logging

logger = logging.getLogger(__name__)

# Thread-safe model state management
_state_lock = threading.Lock()

# Model instance cache for performance optimization
_model_instances = {}
_model_lock = threading.Lock()

MODEL_STATE = {
    'age': {
        'active_version': 4,
        'last_activation': '2025-05-30T03:14:14.881783'
    },
    'content': {
        'active_version': None,
        'last_activation': None
    }
}

# Bu satır Flask'ın dosya değişikliklerini algılaması için
# Her model aktivasyonunda timestamp güncellenir
LAST_UPDATE = "2025-05-30T03:14:14.881783"

def update_model_state(model_type, version_id):
    """
    Thread-safe model state güncelleme
    
    Args:
        model_type (str): Model tipi ('age', 'content', etc.)
        version_id: Versiyon ID'si (int, str, or None)
    """
    global MODEL_STATE, LAST_UPDATE
    
    with _state_lock:
        if model_type not in MODEL_STATE:
            MODEL_STATE[model_type] = {}
        
        MODEL_STATE[model_type]['active_version'] = version_id
        MODEL_STATE[model_type]['last_activation'] = datetime.now().isoformat()
        
        # Flask dosya değişiklik algılaması için timestamp güncelle
        LAST_UPDATE = datetime.now().isoformat()

def get_model_state(model_type=None):
    """
    Thread-safe model state okuma
    
    Args:
        model_type (str, optional): Belirli model tipi, None ise tümü
        
    Returns:
        dict: Model state bilgisi
    """
    with _state_lock:
        if model_type:
            return MODEL_STATE.get(model_type, {}).copy()
        return MODEL_STATE.copy()

def reset_model_state():
    """
    Tüm model state'ini sıfırla
    """
    global MODEL_STATE, LAST_UPDATE
    
    with _state_lock:
        MODEL_STATE = {
            'age': {'active_version': None, 'last_activation': None},
            'content': {'active_version': None, 'last_activation': None}
        }
        LAST_UPDATE = datetime.now().isoformat()

def get_age_estimator():
    """
    Performance-optimized thread-safe singleton InsightFaceAgeEstimator
    
    Returns:
        InsightFaceAgeEstimator: Cached model instance
    """
    cache_key = 'age_estimator'
    
    # Thread-safe cache kontrolü
    with _model_lock:
        if cache_key in _model_instances:
            instance = _model_instances[cache_key]
            if instance is not None:
                logger.debug("Age estimator cache'den kullanılıyor")
                return instance
    
    # Cache miss - yeni instance oluştur
    try:
        from app.ai.insightface_age_estimator import InsightFaceAgeEstimator
        logger.info("Yeni InsightFaceAgeEstimator instance oluşturuluyor...")
        
        start_time = time.time()
        estimator = InsightFaceAgeEstimator()
        load_time = time.time() - start_time
        
        # Thread-safe cache'e kaydet
        with _model_lock:
            _model_instances[cache_key] = estimator
            
        logger.info(f"InsightFaceAgeEstimator cache'e kaydedildi ({load_time:.2f}s)")
        return estimator
        
    except Exception as e:
        logger.error(f"InsightFaceAgeEstimator yükleme hatası: {e}")
        raise

def get_content_analyzer():
    """
    Performance-optimized thread-safe singleton ContentAnalyzer
    
    Returns:
        ContentAnalyzer: Cached model instance
    """
    cache_key = 'content_analyzer'
    
    # Thread-safe cache kontrolü
    with _model_lock:
        if cache_key in _model_instances:
            instance = _model_instances[cache_key]
            if instance is not None and hasattr(instance, 'initialized') and instance.initialized:
                logger.debug("Content analyzer cache'den kullanılıyor")
                return instance
    
    # Cache miss - yeni instance oluştur
    try:
        from app.ai.content_analyzer import ContentAnalyzer
        logger.info("Yeni ContentAnalyzer instance oluşturuluyor...")
        
        start_time = time.time()
        analyzer = ContentAnalyzer()
        load_time = time.time() - start_time
        
        if analyzer.initialized:
            # Thread-safe cache'e kaydet
            with _model_lock:
                _model_instances[cache_key] = analyzer
                
            logger.info(f"ContentAnalyzer cache'e kaydedildi ({load_time:.2f}s)")
            return analyzer
        else:
            raise RuntimeError("ContentAnalyzer initialization failed")
            
    except Exception as e:
        logger.error(f"ContentAnalyzer yükleme hatası: {e}")
        raise

def clear_model_cache(model_type=None):
    """
    Model cache'ini temizle - memory management için
    
    Args:
        model_type (str, optional): Temizlenecek model tipi ('age', 'content'), None ise tümü
    """
    with _model_lock:
        if model_type == 'age':
            if 'age_estimator' in _model_instances:
                instance = _model_instances['age_estimator']
                if hasattr(instance, 'cleanup_models'):
                    instance.cleanup_models()
                del _model_instances['age_estimator']
                logger.info("Age estimator cache temizlendi")
                
        elif model_type == 'content':
            if 'content_analyzer' in _model_instances:
                instance = _model_instances['content_analyzer']
                if hasattr(instance, 'cleanup_models'):
                    instance.cleanup_models()
                del _model_instances['content_analyzer']
                logger.info("Content analyzer cache temizlendi")
                
        else:
            # Tüm cache'i temizle
            for key, instance in _model_instances.items():
                if hasattr(instance, 'cleanup_models'):
                    instance.cleanup_models()
            _model_instances.clear()
            logger.info("Tüm model cache temizlendi")

def get_cache_stats():
    """
    Model cache istatistiklerini döndür
    
    Returns:
        dict: Cache istatistikleri
    """
    with _model_lock:
        stats = {
            'cached_models': list(_model_instances.keys()),
            'cache_size': len(_model_instances),
            'memory_usage': {}
        }
        
        for key, instance in _model_instances.items():
            stats['memory_usage'][key] = {
                'type': type(instance).__name__,
                'initialized': getattr(instance, 'initialized', 'N/A')
            }
            
        return stats