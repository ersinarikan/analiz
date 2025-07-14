import logging
from app.models.content import ContentType

logger = logging.getLogger(__name__)

# Kullanılmayan importlar kaldırıldı

# MIME tiplerine göre içerik türleri
IMAGE_MIME_TYPES = [
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
    'image/bmp', 'image/webp', 'image/tiff', 'image/svg+xml'
]

VIDEO_MIME_TYPES = [
    'video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo', 
    'video/x-ms-wmv', 'video/webm', 'video/x-matroska', 'video/3gpp'
]

def detect_content_type(file_path: str) -> str:
    """
    Dosya yoluna göre içerik türünü tespit eder.
    Args:
        file_path (str): Dosya yolu.
    Returns:
        str: İçerik türü (ör. 'image', 'video', 'unknown').
    """
    if not mime_type:
        return ContentType.UNKNOWN
    
    mime_type = mime_type.lower()
    
    if mime_type in IMAGE_MIME_TYPES:
        return ContentType.IMAGE
    elif mime_type in VIDEO_MIME_TYPES:
        return ContentType.VIDEO
    else:
        return ContentType.UNKNOWN

def check_content_size(file_size: int, content_type: ContentType) -> bool:
    """
    Dosya boyutunun kabul edilebilir sınırlar içinde olup olmadığını kontrol eder.
    
    Args:
        file_size: Dosyanın boyutu (bayt cinsinden)
        content_type: İçerik türü (ContentType enum)
        
    Returns:
        bool: Dosya boyutu kabul edilebilir ise True, değilse False
    """
    # Maksimum dosya boyutları (bayt cinsinden)
    MAX_IMAGE_SIZE = 30 * 1024 * 1024  # 30 MB
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB
    
    if content_type == ContentType.IMAGE:
        return file_size <= MAX_IMAGE_SIZE
    elif content_type == ContentType.VIDEO:
        return file_size <= MAX_VIDEO_SIZE
    else:
        return False

def get_file_extension_for_content_type(content_type: ContentType) -> list[str]:
    """
    İçerik türüne göre kabul edilebilir dosya uzantılarının listesini döndürür.
    
    Args:
        content_type: İçerik türü (ContentType enum)
        
    Returns:
        list: İzin verilen dosya uzantıları listesi
    """
    if content_type == ContentType.IMAGE:
        return ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
    elif content_type == ContentType.VIDEO:
        return ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm', '.3gp', '.mpeg']
    else:
        return []

def validate_file(file_path: str, mime_type: str, file_size: int) -> tuple[bool, ContentType | None, str | None]:
    """
    Dosyayı doğrular, türünü belirler ve boyut kontrolü yapar.
    
    Args:
        file_path: Dosyanın tam yolu
        mime_type: Dosyanın MIME tipi
        file_size: Dosyanın boyutu (bayt cinsinden)
        
    Returns:
        tuple: (başarı durumu, içerik türü, hata mesajı)
    """
    # İçerik türünü belirle
    content_type = detect_content_type(mime_type)
    
    # Desteklenmeyen türdeki dosyaları reddet
    if content_type == ContentType.UNKNOWN:
        return False, None, "Desteklenmeyen dosya türü"
    
    # Boyut kontrolü yap
    if not check_content_size(file_size, content_type):
        max_size = "30 MB" if content_type == ContentType.IMAGE else "500 MB"
        return False, None, f"Dosya çok büyük (maksimum {max_size})"
    
    # Geçerliliği doğrulanmış dosya
    return True, content_type, None 