import sys
import os

# Proje kök dizinini sys.path'e ekle
# Bu script (run_downloader.py) proje kök dizininde çalıştırılıyor.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Flask uygulamasının import edilebilmesi için proje kökünü sys.path'e ekleyebiliriz,
# veya app objesini create_app ile oluşturup context içinde çalışabiliriz.
# Daha temiz bir yol, Flask uygulamasının context'ini kullanmaktır.

from app import create_app # app paketinden create_app'i import et
from app.scripts.download_insightface_model import download_insightface_model
# download_openclip.py içindeki spesifik fonksiyonu import edelim
from download_openclip import download_openclip_ViT_H_14_378_dfn5b

if __name__ == '__main__':
    print("Downloader script'leri Flask uygulama bağlamı olmadan çalıştırılıyor...")
    
    try:
        print("\n--- InsightFace Modeli İndiriliyor/Kontrol Ediliyor ---")
        download_insightface_model()
        print("InsightFace model script'i tamamlandı.")
    except Exception as e:
        print(f"InsightFace runner script'te hata: {e}")

    print("\n-----------------------------------------------------")
    
    try:
        print("\n--- OpenCLIP Modeli (ViT-H-14-378-quickgelu_dfn5b) İndiriliyor/Kontrol Ediliyor ---")
        download_openclip_ViT_H_14_378_dfn5b()
        print("OpenCLIP model script'i tamamlandı.")
    except Exception as e:
        print(f"OpenCLIP runner script'te hata: {e}")

    print("\nTüm indirme script'leri tamamlandı.") 