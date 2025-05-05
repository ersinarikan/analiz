import insightface
import numpy as np
import cv2
import os
import torch
import re
import logging
from config import Config

# Logger oluştur
logger = logging.getLogger(__name__)

# CustomAgeHead sınıfı (train_v1.py'den alınmalı)
class CustomAgeHead(torch.nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Versiyonlu model bulucu fonksiyon
def find_latest_age_model(model_root):
    versions_dir = os.path.join(model_root, 'versions')
    model_filename = 'custom_age_head.pth'
    version_pattern = re.compile(r'^v(\d+)$')
    candidates = []
    
    logger.info(f"Yaş tahmin modeli versiyonları aranıyor: {versions_dir}")
    
    if os.path.exists(versions_dir):
        for name in os.listdir(versions_dir):
            m = version_pattern.match(name)
            if m:
                version_num = int(m.group(1))
                model_path = os.path.join(versions_dir, name, model_filename)
                if os.path.isfile(model_path):
                    candidates.append((version_num, model_path))
                    logger.info(f"Model versiyonu bulundu: v{version_num} - {model_path}")
    
    if candidates:
        candidates.sort(reverse=True)
        latest_version, latest_path = candidates[0]
        logger.info(f"En son model versiyonu seçildi: v{latest_version} - {latest_path}")
        return latest_path
    
    root_model_path = os.path.join(model_root, model_filename)
    if os.path.isfile(root_model_path):
        logger.info(f"Kök dizinde model bulundu: {root_model_path}")
        return root_model_path
    
    logger.warning("Hiçbir model versiyonu bulunamadı!")
    return None

class InsightFaceAgeEstimator:
    def __init__(self, det_size=(640, 640)):
        # Model dosya yolunu ayarla
        model_path = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_l')
        logger.info(f"InsightFaceAgeEstimator başlatılıyor. Model dizini: {model_path}")
        
        # Model dosyalarının varlığını kontrol et
        if not os.path.exists(model_path):
            logger.error(f"Model dosyaları bulunamadı: {model_path}")
            raise FileNotFoundError(f"Model dosyaları bulunamadı: {model_path}")
        
        # Modeli yerel dosyadan yükle
        try:
            self.model = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root=model_path,
                providers=['CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=0, det_size=det_size)
            logger.info("InsightFace temel modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"InsightFace model yükleme hatası: {str(e)}")
            raise
        
        # Kendi yaş modelini yüklemeye çalış
        try:
            age_model_path = find_latest_age_model(model_path)
            if age_model_path is not None:
                logger.info(f"Özel yaş tahmin modeli yükleniyor: {age_model_path}")
                self.age_model = CustomAgeHead()
                self.age_model.load_state_dict(torch.load(age_model_path, map_location='cpu'))
                self.age_model.eval()
                logger.info("Özel yaş tahmin modeli başarıyla yüklendi")
            else:
                logger.warning("Özel yaş tahmin modeli bulunamadı, varsayılan InsightFace yaş tahmini kullanılacak")
                self.age_model = None
        except Exception as e:
            logger.error(f"Özel yaş modeli yükleme hatası: {str(e)}")
            self.age_model = None

    def estimate_age(self, image: np.ndarray):
        """
        Verilen görüntüdeki ilk yüzün yaş tahminini döndürür.
        Args:
            image: BGR (OpenCV) formatında numpy array
        Returns:
            age: float veya None
        """
        faces = self.model.get(image)
        if not faces:
            logger.warning("Görüntüde yüz tespit edilemedi")
            return None
        
        if self.age_model is not None:
            try:
                embedding = faces[0].embedding
                with torch.no_grad():
                    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                    age_pred = self.age_model(emb_tensor).item()
                logger.info(f"Özel model ile yaş tahmini yapıldı: {age_pred:.1f} yaş")
                logger.info(f"[DEBUG] Yaş tahmini başlıyor - face_obj mevcut: {faces[0] is not None}")
                return age_pred
            except Exception as e:
                logger.error(f"Özel model ile yaş tahmini hatası: {str(e)}")
                logger.info("Varsayılan InsightFace yaş tahminine geçiliyor")
                return faces[0].age
        else:
            age = faces[0].age
            logger.info(f"InsightFace ile yaş tahmini yapıldı: {age:.1f} yaş")
            logger.info(f"[DEBUG] Yaş tahmini başlıyor - face_obj mevcut: {faces[0] is not None}")
            return age

    def compute_face_encoding(self, face_image: np.ndarray):
        """
        Verilen yüz görüntüsünden embedding (vektör) çıkarır.
        Args:
            face_image: BGR (OpenCV) formatında numpy array
        Returns:
            embedding: np.ndarray veya None
        """
        faces = self.model.get(face_image)
        if not faces:
            return None
        return faces[0].embedding

    def compare_faces(self, encoding1, encoding2, tolerance=0.6):
        """
        İki embedding (yüz vektörü) arasındaki benzerliği kontrol eder.
        Args:
            encoding1: np.ndarray
            encoding2: np.ndarray
            tolerance: float (daha düşük değer = daha sıkı eşleşme)
        Returns:
            bool: Benzerse True
        """
        if encoding1 is None or encoding2 is None:
            return False
        distance = np.linalg.norm(np.array(encoding1) - np.array(encoding2))
        return distance <= tolerance

# Kullanım örneği:
# estimator = InsightFaceAgeEstimator()
# img = cv2.imread('face.jpg')
# age = estimator.estimate_age(img)
# print('Tahmini yaş:', age) 