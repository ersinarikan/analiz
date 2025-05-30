# Model State File
# Bu dosya model aktivasyonlarında güncellenir ve Flask debug mode tarafından izlenir
# Otomatik restart için config.py tarafından import edilir

import threading
import time
from datetime import datetime

# Thread-safe model state management
_state_lock = threading.Lock()

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