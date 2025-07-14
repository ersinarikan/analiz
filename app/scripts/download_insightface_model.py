import os
import shutil
# sys.path ayarlamaları kaldırıldı

# from app.config import Config # Eski import
from config import Config # config.py kök dizinde olduğu için doğrudan import
import insightface # Bu import burada gerekli, model indirme için.
from app.utils.file_utils import ensure_dir

def download_insightface_model():
    """
    InsightFace 'buffalo_l' modelini indirir ve storage/models/age/buffalo_l/base_model/models/buffalo_l klasörüne kopyalar.
    """
    print("InsightFace 'buffalo_l' modeli (app/scripts altından) indiriliyor...")
    
    # Model klasörünü oluştur
    # Kütüphane root/models/model_adi/ yapısını bekliyor.
    # Root'u '.../base_model' olarak vereceğimiz için, dosyalar '.../base_model/models/buffalo_l/' altında olmalı.
    # Config.MODELS_FOLDER, Config.INSIGHTFACE_AGE_MODEL_NAME vs. kullanılacak
    base_model_root = os.path.join(Config.MODELS_FOLDER, 'age', Config.INSIGHTFACE_AGE_MODEL_NAME, 'base_model')
    target_path_for_onnx_base = os.path.join(base_model_root, 'models', 'buffalo_l')
    ensure_dir(target_path_for_onnx_base)

    # active_model_root = os.path.join(Config.MODELS_FOLDER, 'age', Config.INSIGHTFACE_AGE_MODEL_NAME, 'active_model') # KALDIRILDI
    # target_path_for_onnx_active = os.path.join(active_model_root, 'models', 'buffalo_l') # KALDIRILDI
    # os.makedirs(target_path_for_onnx_active, exist_ok=True) # KALDIRILDI
    
    try:
        print(f"Insightface kütüphanesi {insightface.__version__} kullanılarak model hazırlanıyor...")
        # Modeli indir (bu işlem modeli varsayılan konuma indirecek)
        # providers parametresini FaceAnalysis'ten aldım, prepare'de yok.
        model_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        model_app.prepare(ctx_id=0) # det_size burada verilmiyor.
        
        # Model dosyalarının kaynak konumunu al (genellikle ~/.insightface/models/buffalo_l)
        # insightface.utils.storage.DEFAULT_MODEL_ROOT kullanılabilir mi?
        # Ya da doğrudan ~/.insightface/models/buffalo_l varsayalım.
        source_path = os.path.expanduser('~/.insightface/models/buffalo_l')
        if not os.path.isdir(source_path):
            print(f"HATA: Kaynak model yolu bulunamadı: {source_path}")
            print("InsightFace modeli daha önce hiç indirilmemiş olabilir. Lütfen internet bağlantınızı kontrol edin.")
            return

        print(f"Kaynak model dosyaları {source_path} adresinden kopyalanacak.")
        
        # Model dosyalarını base_model/models/buffalo_l/ altına kopyala
        print(f"'{target_path_for_onnx_base}' hedefi için kopyalanıyor...")
        for file_name in os.listdir(source_path):
            source_file = os.path.join(source_path, file_name)
            target_file = os.path.join(target_path_for_onnx_base, file_name)
            if os.path.isfile(source_file): # Sadece dosyaları kopyala, alt klasörleri değil (eğer varsa)
                shutil.copy2(source_file, target_file)
            
        print(f"Model başarıyla '{target_path_for_onnx_base}' altına kopyalandı.")
        print("İçerik:", os.listdir(target_path_for_onnx_base))

        # Model dosyalarını active_model/models/buffalo_l/ altına da kopyala - BU BÖLÜM KALDIRILDI
        # print(f"'{target_path_for_onnx_active}' hedefi için kopyalanıyor...")
        # if os.path.exists(target_path_for_onnx_base): 
        #     for item_name in os.listdir(target_path_for_onnx_base):
        #         s = os.path.join(target_path_for_onnx_base, item_name)
        #         d = os.path.join(target_path_for_onnx_active, item_name)
        #         if os.path.isfile(s):
        #              shutil.copy2(s, d)
        #     print(f"Base model dosyaları ayrıca şuraya kopyalandı: {target_path_for_onnx_active}")
        #     print(f"Active model (models/buffalo_l) klasör içeriği: {os.listdir(target_path_for_onnx_active)}")
        # else:
        #     print(f"Kaynak base model ({target_path_for_onnx_base}) bulunamadı, active_model'e kopyalama yapılamadı.")
        
        print("İndirme ve base_model'e kopyalama tamamlandı.")

    except Exception as e:
        print(f"Model indirme/kopyalama hatası: {str(e)}")
        # raise # Hata durumunda scriptin durmaması için raise'ı kaldırabiliriz veya loglayabiliriz.

# Bu script doğrudan çalıştırıldığında download_insightface_model fonksiyonunu çağırır.
# python -m app.scripts.download_insightface_model olarak çağrılacak.
# if __name__ == '__main__': 
# Bu bloğa gerek yok eğer modül olarak çalıştırılacaksa. 
# Ama doğrudan python app/scripts/download_insightface_model.py denemesi için kalabilir.
# Ancak göreceli importlar __main__ ile çalışmaz.
# Bu yüzden bu scripti __main__ olarak çalıştırmamalıyız. 