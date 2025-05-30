import os
from flask import current_app

# Proje kök dizinini belirle (ör: WSANALIZ klasörü) - workspace root olmalı
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Go up 2 levels from app/utils/ to workspace root

def to_rel_path(abs_path):
    """Mutlak yolu, workspace köküne göre bağıl yapar ve slash normalize eder."""
    if not abs_path:
        return ""
    
    try:
        rel_path = os.path.relpath(abs_path, BASE_DIR)
        # "../" ile başlamasını önle - eğer workspace dışındaysa absolute path döndür  
        if rel_path.startswith('..'):
            return abs_path.replace("\\", "/")
        return rel_path.replace("\\", "/")
    except ValueError:
        # farklı drive'larda olabilir (Windows)
        return abs_path.replace("\\", "/")

def to_abs_path(rel_path):
    """Bağıl yolu, proje köküne göre mutlak yapar."""
    return os.path.abspath(os.path.join(BASE_DIR, rel_path))

def is_subpath(path, base=BASE_DIR):
    """Bir path'in base'in altında olup olmadığını kontrol eder (güvenlik için)."""
    return os.path.commonpath([os.path.abspath(path), base]) == base 