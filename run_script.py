import sys
import os
import runpy
import logging

# Basit bir logging ayarı
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(f"run_script.py başlatıldı. Argümanlar: {sys.argv}")
    if len(sys.argv) < 2:
        logger.error("Kullanım hatası: Çalıştırılacak betik yolu belirtilmedi.")
        print("Kullanım: python run_script.py <çalıştırılacak_betik_yolu> [betik_argümanları...]")
        print("Örnek: python run_script.py app/scripts/benim_betigim.py --parametre deger")
        sys.exit(1)

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        logger.debug(f"Proje kökü ({project_root}) sys.path'e ekleniyor.")
        sys.path.insert(0, project_root)
    else:
        logger.debug(f"Proje kökü ({project_root}) zaten sys.path içinde.")
    
    script_to_run = sys.argv[1]
    logger.info(f"Çalıştırılacak betik: {script_to_run}")
    
    original_sys_argv = list(sys.argv)
    sys.argv = [script_to_run] + original_sys_argv[2:]
    logger.debug(f"Hedef betik için ayarlanmış sys.argv: {sys.argv}")
        
    try:
        logger.info(f"runpy.run_path ile '{script_to_run}' çalıştırılıyor...")
        runpy.run_path(script_to_run, run_name="__main__")
        logger.info(f"'{script_to_run}' başarıyla tamamlandı.")
    except SystemExit as e:
        logger.warning(f"'{script_to_run}' betiği SystemExit ile sonlandı (çıkış kodu: {e.code}). Bu normal bir çıkış olabilir.")
        # SystemExit durumunda orijinal sys.exit kodunu koru, eğer varsa
        if e.code is not None:
            sys.exit(e.code)
    except Exception as e:
        logger.error(f"'{script_to_run}' betiği çalıştırılırken genel bir hata oluştu: {e}", exc_info=True)
        sys.exit(1)
    finally:
        sys.argv = original_sys_argv
        logger.debug("Orijinal sys.argv geri yüklendi.") 