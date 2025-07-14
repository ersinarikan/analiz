import subprocess
import sys
import os
import logging
import threading
import time
import signal

logger = logging.getLogger(__name__)

PID_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wsanaliz.pid')

# PID dosyası işlemleri
def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def read_pid():
    try:
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def remove_pid():
    try:
        os.remove(PID_FILE)
    except Exception:
        pass

def stop_app():
    if sys.platform == "win32":
        # Komut satırında 'app.py' geçen tüm python.exe süreçlerini bul ve öldür
        import subprocess
        result = subprocess.check_output('wmic process where "CommandLine like \'%app.py%\' and Name=\'python.exe\'" get ProcessId', shell=True)
        lines = result.decode().splitlines()
        pids = [line.strip() for line in lines if line.strip().isdigit()]
        if not pids:
            print("Kapatılacak süreç bulunamadı.")
        for pid in pids:
            try:
                subprocess.call(["taskkill", "/PID", pid, "/F"])
                print(f"PID {pid} sonlandırıldı.")
            except Exception as e:
                print(f"PID {pid} sonlandırılamadı: {e}")
    else:
        # Linux/Mac: PID dosyasını kullan
        pid = read_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"PID {pid} sonlandırıldı.")
            except Exception as e:
                print(f"PID sonlandırılamadı: {e}")
        else:
            print("PID dosyası bulunamadı veya geçersiz.")

def start_app():
    subprocess.Popen([sys.executable, "app.py"])
    print("Yeni uygulama başlatıldı.")

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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "stop":
            stop_app()
        elif sys.argv[1] == "start":
            start_app()
        else:
            print("Kullanım: python -m app.utils.restart_helper [stop|start]")
    else:
        print("Kullanım: python -m app.utils.restart_helper [stop|start]") 