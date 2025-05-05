import insightface
import os
import shutil
from config import Config

def download_insightface_model():
    """
    InsightFace 'buffalo_l' modelini indirir ve storage/models/age klasörüne kopyalar.
    """
    print("InsightFace 'buffalo_l' modeli indiriliyor...")
    
    # Model klasörünü oluştur
    target_path = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_l')
    os.makedirs(target_path, exist_ok=True)
    
    try:
        # Modeli indir (bu işlem modeli varsayılan konuma indirecek)
        model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0)
        
        # Model dosyalarının kaynak konumunu al
        source_path = os.path.expanduser('~/.insightface/models/buffalo_l')
        
        # Model dosyalarını kopyala
        for file in os.listdir(source_path):
            source_file = os.path.join(source_path, file)
            target_file = os.path.join(target_path, file)
            shutil.copy2(source_file, target_file)
            
        print(f"Model başarıyla indirildi ve kopyalandı: {target_path}")
        
    except Exception as e:
        print(f"Model indirme/kopyalama hatası: {str(e)}")
        raise

if __name__ == "__main__":
    download_insightface_model() 