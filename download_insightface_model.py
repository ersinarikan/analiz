import insightface
import os
import shutil
from config import Config

def download_insightface_model():
    """
    InsightFace buffalo modelini indirir ve storage/models/age klasörüne buffalo_x adıyla kopyalar.
    """
    print("InsightFace modeli buffalo_x olarak indiriliyor...")
    
    # Kullanılacak model adı (mevcut olduğu bilinen)
    source_model_name = "buffalo_sc"  # Gerçekte kullanacağımız model
    target_model_name = "buffalo_x"   # Hedef klasör adı
    
    # Model klasörünü oluştur
    target_path = os.path.join(Config.MODELS_FOLDER, 'age', target_model_name)
    os.makedirs(target_path, exist_ok=True)
    
    try:
        # Modeli indir (bu işlem modeli varsayılan konuma indirecek)
        print(f"Şu anda mevcut olan {source_model_name} modelini indiriyoruz...")
        model = insightface.app.FaceAnalysis(name=source_model_name, providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0)
        
        # Model dosyalarının kaynak konumunu al
        source_path = os.path.expanduser(f'~/.insightface/models/{source_model_name}')
        
        # Model dosyalarını kopyala
        for file in os.listdir(source_path):
            source_file = os.path.join(source_path, file)
            target_file = os.path.join(target_path, file)
            shutil.copy2(source_file, target_file)
            
        print(f"Model başarıyla indirildi ve '{target_model_name}' olarak kopyalandı: {target_path}")
        print("NOT: Gerçek buffalo_x modeli henüz indirilemediğinden buffalo_sc kullanılmaktadır.")
        
    except Exception as e:
        print(f"Model indirme/kopyalama hatası: {str(e)}")
        raise

if __name__ == "__main__":
    download_insightface_model() 