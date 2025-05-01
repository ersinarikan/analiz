import os
import shutil
import time
from pathlib import Path
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_remove(path, retries=3, delay=1):
    """
    Dosya veya klasörü güvenli bir şekilde silmeye çalışır.
    Windows'ta izin hatalarını önlemek için birkaç kez dener.
    
    Args:
        path: Silinecek dosya/klasör yolu
        retries: Deneme sayısı
        delay: Denemeler arası bekleme süresi (saniye)
    """
    for i in range(retries):
        try:
            if os.path.isfile(path):
                os.chmod(path, 0o777)  # Tam izin ver
                os.remove(path)
                logger.info(f"Dosya silindi: {path}")
                return True
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Klasör silindi: {path}")
                return True
        except Exception as e:
            logger.warning(f"Silme denemesi {i+1}/{retries} başarısız: {str(e)}")
            if i < retries - 1:
                time.sleep(delay)
    return False

def cleanup_project():
    """
    Proje klasörünü temizler.
    """
    # Temizlenecek klasörler ve dosyalar
    cleanup_items = [
        # Python önbellek dosyaları
        "__pycache__",
        "**/__pycache__",
        "**/**/__pycache__",
        
        # Geçici dosyalar
        "instance",
        "storage/temp",
        "storage/uploads",
        "test_images",
        
        # Diğer geçici dosyalar
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".coverage",
        "htmlcov",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache"
    ]
    
    # Proje kök dizini
    root_dir = Path(__file__).parent.parent
    
    # Temizleme işlemi
    for pattern in cleanup_items:
        for path in root_dir.glob(pattern):
            if path.exists():
                logger.info(f"Temizleniyor: {path}")
                safe_remove(path)
    
    # Özel klasörler için ek kontroller
    storage_dir = root_dir / "storage"
    if storage_dir.exists():
        # storage klasörünü temizle ama klasörün kendisini silme
        for item in storage_dir.iterdir():
            if item.is_file():
                safe_remove(item)
            elif item.is_dir() and item.name != "models":  # models klasörünü koru
                safe_remove(item)
    
    logger.info("Temizleme işlemi tamamlandı!")

if __name__ == "__main__":
    try:
        cleanup_project()
    except Exception as e:
        logger.error(f"Temizleme sırasında hata oluştu: {str(e)}") 