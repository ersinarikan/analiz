import subprocess
import sys
import os
import logging
import threading
import time

logger = logging.getLogger(__name__)

def restart_application(delay=1):
    """
    Platform bağımsız şekilde uygulamayı yeniden başlatır.
    
    Args:
        delay (int): Yeniden başlatma öncesi bekleme süresi (saniye)
    
    Returns:
        bool: Başarılı olup olmadığını belirtir
    """
    try:
        # Mevcut dosya yolunu al
        current_path = os.path.dirname(os.path.abspath(__file__))
        # app/utils -> app -> project_root
        project_root = os.path.dirname(os.path.dirname(current_path))
        app_path = os.path.join(project_root, 'app.py')
        
        logger.info(f"Uygulama yeniden başlatılıyor... Root: {project_root}")
        
        # Platform bağımsız subprocess parametreleri
        subprocess_kwargs = {
            'cwd': project_root,
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.DEVNULL,
            'stdin': subprocess.DEVNULL
        }
        
        # Platform spesifik ayarlar
        if sys.platform == 'win32':
            # Windows: Konsol penceresi açmadan çalıştır
            subprocess_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            # Unix/Linux: Yeni session başlat (daemon-like)
            subprocess_kwargs['start_new_session'] = True
        
        # Yeni süreç başlat
        new_process = subprocess.Popen([
            sys.executable, app_path
        ], **subprocess_kwargs)
        
        logger.info(f"Yeni süreç başlatıldı (PID: {new_process.pid})")
        
        # Gecikmeli shutdown başlat
        def delayed_shutdown():
            time.sleep(delay)
            logger.info("Mevcut süreç kapatılıyor...")
            os._exit(0)
        
        shutdown_thread = threading.Thread(target=delayed_shutdown)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        
        return True
        
    except subprocess.SubprocessError as e:
        logger.error(f"Subprocess hatası: {str(e)}")
        # Hata durumunda basit restart dene
        try:
            subprocess.Popen([
                sys.executable, 
                os.path.join(project_root, 'app.py')
            ], cwd=project_root)
            return True
        except Exception as fallback_error:
            logger.error(f"Fallback restart hatası: {str(fallback_error)}")
            return False
            
    except Exception as e:
        logger.error(f"Beklenmeyen restart hatası: {str(e)}")
        return False

def restart_for_model_change(model_type, delay=1):
    """
    Model değişikliği sonrası sistemi yeniden başlatır.
    
    Args:
        model_type (str): Değişen model tipi ('age', 'content')
        delay (int): Yeniden başlatma öncesi bekleme süresi
    
    Returns:
        bool: Başarılı olup olmadığını belirtir
    """
    logger.info(f"Model değişikliği nedeniyle restart: {model_type}")
    return restart_application(delay)

def restart_for_parameter_change(delay=1):
    """
    Parametre değişikliği sonrası sistemi yeniden başlatır.
    
    Args:
        delay (int): Yeniden başlatma öncesi bekleme süresi
    
    Returns:
        bool: Başarılı olup olmadığını belirtir
    """
    logger.info("Parametre değişikliği nedeniyle restart")
    return restart_application(delay)

def is_windows():
    """Windows platformu kontrolü"""
    return sys.platform == 'win32'

def is_unix():
    """Unix/Linux platformu kontrolü"""
    return sys.platform in ['linux', 'darwin', 'freebsd'] 