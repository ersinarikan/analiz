import os
import shutil
import uuid
import mimetypes
import json
import logging
from flask import current_app
from werkzeug.utils import secure_filename

def is_allowed_file(filename: str) -> bool:
    """
    Dosya adının izin verilen uzantıya sahip olup olmadığını kontrol eder.
    Args:
        filename (str): Dosya adı.
    Returns:
        bool: İzin veriliyorsa True, aksi halde False.
    """
    if allowed_extensions is None:
        # Varsayılan izin verilen uzantılar
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'}
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_file_extension(filename):
    """Dosya uzantısını döndürür."""
    return os.path.splitext(filename)[1].lower()

def generate_unique_filename(original_filename):
    """Benzersiz bir dosya adı oluşturur."""
    # Güvenli bir dosya adı oluştur
    filename = secure_filename(original_filename)
    
    # Uzantıyı al
    ext = get_file_extension(filename)
    
    # Benzersiz UUID oluştur
    unique_id = str(uuid.uuid4())
    
    # Benzersiz dosya adını oluştur
    unique_filename = f"{unique_id}{ext}"
    
    return unique_filename

def get_file_mimetype(file_path):
    """Dosyanın MIME türünü döndürür."""
    try:
        # mimetypes kütüphanesini kullan
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # Fallback to common types based on extension
            ext = file_path.rsplit('.', 1)[1].lower() if '.' in file_path else ''
            if ext in ['jpg', 'jpeg']:
                mime_type = 'image/jpeg'
            elif ext == 'png':
                mime_type = 'image/png'
            elif ext == 'gif':
                mime_type = 'image/gif'
            elif ext == 'mp4':
                mime_type = 'video/mp4'
            elif ext == 'avi':
                mime_type = 'video/x-msvideo'
            elif ext == 'mov':
                mime_type = 'video/quicktime'
            else:
                mime_type = 'application/octet-stream'
        return mime_type
    except:
        return 'application/octet-stream'

def is_image_file(mime_type):
    """Dosyanın bir resim dosyası olup olmadığını kontrol eder."""
    return mime_type.startswith('image/')

def is_video_file(mime_type):
    """Dosyanın bir video dosyası olup olmadığını kontrol eder."""
    return mime_type.startswith('video/')

def get_file_size(file_path):
    """Dosyanın boyutunu bayt cinsinden döndürür."""
    return os.path.getsize(file_path)

def create_directory(directory_path):
    """Belirtilen dizini oluşturur (yoksa)."""
    os.makedirs(directory_path, exist_ok=True)
    return os.path.exists(directory_path)

def move_file(source_path: str, destination_path: str) -> bool:
    """Bir dosyayı kaynaktan hedefe taşır."""
    try:
        # Hedef dizinin var olduğundan emin ol
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Dosyayı taşı
        shutil.move(source_path, destination_path)
        
        return os.path.exists(destination_path)
    
    except Exception as e:
        current_app.logger.error(f"Dosya taşıma hatası: {str(e)}")
        return False

def copy_file(source_path: str, destination_path: str) -> bool:
    """Bir dosyayı kaynaktan hedefe kopyalar."""
    try:
        # Hedef dizinin var olduğundan emin ol
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Dosyayı kopyala
        shutil.copy2(source_path, destination_path)
        
        return os.path.exists(destination_path)
    
    except Exception as e:
        current_app.logger.error(f"Dosya kopyalama hatası: {str(e)}")
        return False

def delete_file(file_path: str) -> bool:
    """Belirtilen dosyayı siler."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return not os.path.exists(file_path)
    
    except Exception as e:
        current_app.logger.error(f"Dosya silme hatası: {str(e)}")
        return False

def get_file_list(directory_path: str, filter_func=None):
    """Belirtilen dizindeki dosyaları listeler."""
    try:
        if not os.path.exists(directory_path):
            return []
        
        files = []
        
        # Dizindeki dosyaları listele
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Sadece dosyaları kontrol et (dizinleri atla)
            if os.path.isfile(file_path):
                # Eğer bir filtre fonksiyonu belirtilmişse uygula
                if filter_func is None or filter_func(file_path):
                    files.append(file_path)
        
        return files
    
    except Exception as e:
        current_app.logger.error(f"Dosya listeleme hatası: {str(e)}")
        return []

def get_media_files(directory_path: str):
    """Belirtilen dizindeki resim ve video dosyalarını listeler."""
    def is_media_file(file_path):
        mime_type = get_file_mimetype(file_path)
        return is_image_file(mime_type) or is_video_file(mime_type)
    
    return get_file_list(directory_path, is_media_file)

def format_file_size(size_bytes: int) -> str:
    """Bayt cinsinden dosya boyutunu okunabilir bir formata dönüştürür."""
    # İkinin katları için bölme katsayısı (1024)
    factor = 1024
    
    # Birimleri tanımla
    size_units = ["B", "KB", "MB", "GB", "TB", "PB"]
    
    # Boyut 1 bayttan küçükse, 0 bayt döndür
    if size_bytes < 1:
        return "0 B"
    
    # Uygun birimi bul
    i = 0
    while size_bytes >= factor and i < len(size_units) - 1:
        size_bytes /= factor
        i += 1
    
    # Sonucu formatla (1 decimal precision)
    return f"{size_bytes:.1f} {size_units[i]}" 

def ensure_dir(path):
    """Dizini yoksa oluşturur."""
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Dizin oluşturuldu veya zaten mevcut: {path}")
    except Exception as e:
        logging.error(f"Dizin oluşturulurken hata: {path} - {e}")
        raise

def safe_copytree(src, dst):
    """Varışta varsa silip src'den dst'ye kopyalar."""
    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)
            logging.info(f"Varış dizini silindi: {dst}")
        shutil.copytree(src, dst)
        logging.info(f"Kopyalandı: {src} -> {dst}")
    except Exception as e:
        logging.error(f"Kopyalama hatası: {src} -> {dst} - {e}")
        raise

def safe_remove(path):
    """Dosya veya dizini güvenli şekilde siler."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            logging.info(f"Dizin silindi: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            logging.info(f"Dosya silindi: {path}")
    except Exception as e:
        logging.error(f"Silme hatası: {path} - {e}")
        raise

def write_json(path, data):
    """Verilen veriyi JSON olarak yazar."""
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        logging.info(f"JSON dosyası yazıldı: {path}")
    except Exception as e:
        logging.error(f"JSON yazma hatası: {path} - {e}")
        raise

def read_json(path):
    """JSON dosyasını okur ve döndürür."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logging.info(f"JSON dosyası okundu: {path}")
        return data
    except Exception as e:
        logging.error(f"JSON okuma hatası: {path} - {e}")
        raise

def get_folder_size(path):
    """Dizinin boyutunu MB cinsinden döndürür."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        size_mb = round(total_size / (1024 * 1024), 2)
        logging.info(f"Dizin boyutu hesaplandı: {path} = {size_mb} MB")
        return size_mb
    except Exception as e:
        logging.error(f"Dizin boyutu hesaplanırken hata: {path} - {e}")
        return 0 