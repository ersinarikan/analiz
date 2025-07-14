import os

# Kullanılmayan importlar kaldırıldı

# Proje kök dizinini belirle (ör: WSANALIZ klasörü) - workspace root olmalı
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Go up 2 levels from app/utils/ to workspace root

def to_rel_path(abs_path: str) -> str:
    """
    Mutlak dosya yolunu göreli yola çevirir.
    Args:
        abs_path (str): Mutlak dosya yolu.
    Returns:
        str: Göreli dosya yolu.
    """
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

def to_abs_path(rel_path: str) -> str:
    """Bağıl yolu, proje köküne göre mutlak yapar."""
    return os.path.abspath(os.path.join(BASE_DIR, rel_path))

def is_subpath(path: str, base: str = BASE_DIR) -> bool:
    """Bir path'in base'in altında olup olmadığını kontrol eder (güvenlik için)."""
    return os.path.commonpath([os.path.abspath(path), base]) == base 