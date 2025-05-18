import os
import sys

# Çalıştırıldığı dizini (proje kök dizini olmalı) Python arama yoluna ekle
current_working_directory = os.getcwd() # Mevcut çalışma dizinini al
if current_working_directory not in sys.path:
    sys.path.insert(0, current_working_directory)

import shutil
from app.config import Config

def download_insightface_model():
    """
    InsightFace 'buffalo_l' modelini indirir ve storage/models/age/buffalo_l/base_model klasörüne kopyalar.
    """
    print("InsightFace 'buffalo_l' modeli indiriliyor...")
    
    # Model klasörünü oluştur
    # target_path = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_l', 'base_model') # Eski
    # Kütüphane root/models/model_adi/ yapısını bekliyor.
    # Root'u '.../base_model' olarak vereceğimiz için, dosyalar '.../base_model/models/buffalo_l/' altında olmalı.
    target_path_base_for_onnx = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_l', 'base_model', 'models', 'buffalo_l')
    os.makedirs(target_path_base_for_onnx, exist_ok=True)
    
    try:
        # Modeli indir (bu işlem modeli varsayılan konuma indirecek)
        model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0)
        
        # Model dosyalarının kaynak konumunu al
        source_path = os.path.expanduser('~/.insightface/models/buffalo_l')
        
        # Model dosyalarını kopyala
        for file in os.listdir(source_path):
            source_file = os.path.join(source_path, file)
            # target_file = os.path.join(target_path, file) # Eski
            target_file = os.path.join(target_path_base_for_onnx, file) # Yeni
            shutil.copy2(source_file, target_file)
            
        print(f"Model başarıyla indirildi ve kopyalandı: {target_path_base_for_onnx}")
        print(f"Model dosyaları şuraya kopyalandı: {target_path_base_for_onnx}")
        print("İçerik:", os.listdir(target_path_base_for_onnx))

        # base_model içeriğini active_model klasörüne de kopyala
        # active_model_path = os.path.join(Config.MODELS_FOLDER, 'age', Config.INSIGHTFACE_AGE_MODEL_NAME, 'active_model') # Eski
        active_model_path_base_for_onnx = os.path.join(Config.MODELS_FOLDER, 'age', Config.INSIGHTFACE_AGE_MODEL_NAME, 'active_model', 'models', 'buffalo_l')
        os.makedirs(active_model_path_base_for_onnx, exist_ok=True)
        
        # if os.path.exists(target_path): # Eski
        if os.path.exists(target_path_base_for_onnx): # Yeni
            # for item_name in os.listdir(target_path): # Eski
            for item_name in os.listdir(target_path_base_for_onnx): # Yeni
                # s = os.path.join(target_path, item_name) # Eski
                s = os.path.join(target_path_base_for_onnx, item_name) # Yeni
                # d = os.path.join(active_model_path, item_name) # Eski
                d = os.path.join(active_model_path_base_for_onnx, item_name) # Yeni
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print(f"Base model dosyaları ayrıca şuraya kopyalandı: {active_model_path_base_for_onnx}")
            print(f"Active model klasör içeriği: {os.listdir(active_model_path_base_for_onnx)}")
        else:
            # print(f"Kaynak base_model yolu ({target_path}) bulunamadı, active_model'e kopyalama yapılamadı.") # Eski
            print(f"Kaynak base_model yolu ({target_path_base_for_onnx}) bulunamadı, active_model'e kopyalama yapılamadı.") # Yeni
        
    except Exception as e:
        print(f"Model indirme/kopyalama hatası: {str(e)}")
        raise

if __name__ == "__main__":
    download_insightface_model() 