import os
import uuid
import mimetypes
from flask import current_app
from werkzeug.utils import secure_filename
from PIL import Image
import io
from config import Config
import cv2
import numpy as np
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# İzin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'}

def is_allowed_file(filename):
    """
    Dosya uzantısının izin verilen uzantılar listesinde olup olmadığını kontrol eder.
    
    Args:
        filename: Kontrol edilecek dosyanın adı
        
    Returns:
        bool: Dosya uzantısı izin veriliyorsa True, değilse False
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def get_file_mimetype(filename):
    """
    Dosyanın MIME tipini döndürür.
    
    Args:
        filename: Dosyanın tam yolu
        
    Returns:
        str: Dosyanın MIME tipi
    """
    try:
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            # Fallback to common types based on extension
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
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
    except Exception as e:
        current_app.logger.error(f"MIME tipi belirlenirken hata: {str(e)}")
        return None

def save_uploaded_file(file, filename=None):
    """
    Yüklenen dosyayı güvenli bir şekilde kaydeder.
    
    Args:
        file: Yüklenen dosya nesnesi
        filename: Dosya için özel isim (belirtilmezse otomatik oluşturulur)
        
    Returns:
        str: Kaydedilen dosyanın yolu
    """
    try:
        if filename is None:
            # Güvenli ve benzersiz bir dosya adı oluştur
            original_filename = secure_filename(file.filename)
            ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
            filename = f"{uuid.uuid4().hex}.{ext}"
        
        # Dosya yolunu oluştur
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        # Dizinin var olduğundan emin ol
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Dosyayı kaydet
        file.save(file_path)
        
        return file_path
    
    except Exception as e:
        current_app.logger.error(f"Dosya kaydedilirken hata: {str(e)}")
        return None

def get_file_info(file_path):
    """
    Dosya hakkında temel bilgileri alır.
    
    Args:
        file_path: Bilgileri alınacak dosyanın tam yolu
        
    Returns:
        dict: Dosya bilgilerini içeren sözlük veya dosya bulunamazsa None
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Dosya bulunamadı: {file_path}")
            return None
        
        file_size = os.path.getsize(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if not mime_type:
            # MIME tipi belirlenemezse, 'application/octet-stream' varsayalım
            mime_type = 'application/octet-stream'
        
        return {
            'path': file_path,
            'size': file_size,
            'mime_type': mime_type
        }
    except Exception as e:
        logger.error(f"Dosya bilgisi alınırken hata oluştu: {str(e)}")
        return None

def create_thumbnail(file_path, mime_type, max_size=(200, 200)):
    """
    Dosya için küçük resim (thumbnail) oluşturur.
    
    Args:
        file_path: Küçük resmi oluşturulacak dosyanın tam yolu
        mime_type: Dosyanın MIME tipi
        max_size: Küçük resmin maksimum boyutu (genişlik, yükseklik)
        
    Returns:
        bytes: Küçük resim verisi veya oluşturulamazsa None
    """
    try:
        if mime_type.startswith('image/'):
            # Resim dosyası için küçük resim oluştur
            img = Image.open(file_path)
            img.thumbnail(max_size)
            
            # BytesIO nesnesi oluştur ve resmi kaydet
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
            
        elif mime_type.startswith('video/'):
            # Video dosyası için küçük resim oluştur (ilk kare)
            video = cv2.VideoCapture(file_path)
            success, frame = video.read()
            
            if success:
                # OpenCV BGR formatından RGB'ye çevir
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail(max_size)
                
                # BytesIO nesnesi oluştur ve resmi kaydet
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                return buffer.getvalue()
            else:
                return None
        else:
            # Desteklenmeyen MIME tipi
            return None
    except Exception as e:
        logger.error(f"Küçük resim oluşturulurken hata oluştu: {str(e)}")
        return None

def save_upload_file(uploaded_file, filename):
    """
    Yüklenen dosyayı kaydeder.
    
    Args:
        uploaded_file: Yüklenen dosya nesnesi (genellikle flask.request.files içindeki dosya)
        filename: Kaydedilecek dosya adı
        
    Returns:
        str: Kaydedilen dosyanın tam yolu veya hata durumunda None
    """
    try:
        # Yükleme klasörünü al
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        # Dosyayı kaydet
        file_path = os.path.join(upload_folder, filename)
        uploaded_file.save(file_path)
        
        return file_path
    except Exception as e:
        logger.error(f"Dosya kaydedilirken hata oluştu: {str(e)}")
        return None

def get_file_extension(filename):
    """
    Dosya adından uzantıyı çıkarır.
    
    Args:
        filename: Uzantısı alınacak dosya adı
        
    Returns:
        str: Dosya uzantısı (nokta ile birlikte) veya boş string
    """
    try:
        return os.path.splitext(filename)[1].lower()
    except:
        return ""

def is_allowed_file(filename, allowed_extensions):
    """
    Dosya uzantısının izin verilen uzantılar listesinde olup olmadığını kontrol eder.
    
    Args:
        filename: Kontrol edilecek dosya adı
        allowed_extensions: İzin verilen uzantıların listesi
        
    Returns:
        bool: Dosya uzantısı izin veriliyorsa True, değilse False
    """
    return get_file_extension(filename) in allowed_extensions

def delete_file(file_path):
    """
    Dosyayı siler.
    
    Args:
        file_path: Silinecek dosyanın tam yolu
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        logger.error(f"Dosya silinirken hata oluştu: {str(e)}")
        return False 