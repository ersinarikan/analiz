import insightface
import numpy as np
import cv2
import os
import torch
import re
import logging
from config import Config
import clip  # CLIP'i import ediyoruz
from PIL import Image  # PIL kütüphanesini ekliyoruz
import math
from flask import current_app # current_app import edildi

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
def find_latest_age_model(model_path):
    age_model_dir = os.path.join(model_path, 'models') # veya doğrudan model_path altında olabilir
    if not os.path.isdir(age_model_dir):
        logger.warning(f"Yaş modeli için 'models' klasörü bulunamadı: {age_model_dir}")
        return None
    
    model_files = [f for f in os.listdir(age_model_dir) if f.startswith('age_model_epoch_') and f.endswith('.pth')]
    if not model_files:
        logger.warning(f"'models' klasöründe özel yaş modeli bulunamadı: {age_model_dir}")
        return None
    
    # Epoch numarasına göre sırala ve en sonuncuyu al
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    latest_model_file = model_files[0]
    logger.info(f"Bulunan en son özel yaş modeli: {latest_model_file}")
    return os.path.join(age_model_dir, latest_model_file)

class InsightFaceAgeEstimator:
    def __init__(self, det_size=(640, 640)):
        # Model dosya yolunu ayarla
        # Config.MODELS_FOLDER yerine current_app.config['MODELS_FOLDER'] kullanmak daha tutarlı olabilir
        # Ancak Config importu zaten yapılmış ve bu şekilde kullanılıyor, şimdilik dokunmuyoruz.
        model_path = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_l')
        logger.info(f"InsightFaceAgeEstimator başlatılıyor. Model dizini: {model_path}")
        
        # Model dosyalarının varlığını kontrol et
        if not os.path.exists(model_path):
            logger.error(f"Model dosyaları bulunamadı: {model_path}")
            raise FileNotFoundError(f"Model dosyaları bulunamadı: {model_path}")
        
        # FACE_DETECTION_CONFIDENCE değerini config'den oku
        # FACTORY_DEFAULTS'taki değer 0.5, kodda kullanılan 0.2 idi.
        # Config'den gelen değer öncelikli olacak.
        face_detection_thresh = current_app.config.get('FACE_DETECTION_CONFIDENCE', 0.5) 
        logger.info(f"Kullanılacak yüz tespit eşiği (det_thresh): {face_detection_thresh}")

        # Modeli yerel dosyadan yükle
        try:
            self.model = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root=model_path,
                providers=['CPUExecutionProvider'],
                det_thresh=face_detection_thresh # Dinamik olarak okunan değeri kullan
            )
            self.model.prepare(ctx_id=0, det_size=det_size)
            logger.info(f"InsightFace temel modeli başarıyla yüklendi (det_thresh={face_detection_thresh} ile)")
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
            
        # CLIP modelini yükle
        try:
            logger.info("CLIP modeli yükleniyor (yaş tahmin güven skoru için)")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Sadece büyük modeli yükle, küçük modele otomatik geçiş yok
            logger.info("ViT-B/32 CLIP modeli yükleniyor - en büyük ve doğru model")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            logger.info(f"CLIP modeli ({device} üzerinde) başarıyla yüklendi.")
            
            self.clip_device = device
            logger.info(f"CLIP modeli başarıyla yüklendi, çalışma ortamı: {device}")
        except Exception as e:
            logger.error(f"CLIP modeli yüklenemedi: {str(e)}")
            logger.warning("CLIP modeli olmadan güven skoru 0.5 olarak sabitlenecek")
            self.clip_model = None
            self.clip_preprocess = None

    def estimate_age(self, full_image: np.ndarray, face):
        """
        Verilen 'face' nesnesi için yaş tahminini ve CLIP güven skorunu döndürür.
        Yüz tespiti bu fonksiyonda *yapılmaz*, önceden tespit edilmiş face nesnesi kullanılır.

        Args:
            full_image (np.ndarray): Yüzün bulunduğu orijinal tam kare (BGR).
            face: InsightFace modelinin get() metodundan dönen yüz nesnesi.

        Returns:
            Tuple: (estimated_age, confidence_score) veya (None, None - hata durumunda)
        """
        if face is None:
            logger.warning("estimate_age: Geçersiz 'face' nesnesi alındı (None). Varsayılan değerler dönülüyor.")
            return 25.0, 0.5

        logger.info(f"[AGE_LOG] estimate_age başladı. Gelen face bbox: {face.bbox}, Ham Yaş: {face.age}")
        raw_insightface_age = face.age

        # Yaş değeri None ise varsayılanı kullan
        if raw_insightface_age is None:
            logger.warning("[AGE_LOG] InsightFace ham yaşı None, varsayılan (25) kullanılacak.")
            estimated_age = 25
        else:
            # Özel yaş modeli kontrolü
            if self.age_model is not None:
                try:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        logger.warning("[AGE_LOG] Özel model için embedding yok, InsightFace yaşı ({raw_insightface_age}) kullanılacak.")
                        estimated_age = raw_insightface_age
                    else:
                        embedding = face.embedding
                        with torch.no_grad():
                            emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                            age_pred = self.age_model(emb_tensor).item()
                        logger.info(f"[AGE_LOG] Özel yaş modeli tahmini: {age_pred:.1f}. Bu kullanılacak.")
                        estimated_age = age_pred
                except Exception as e:
                    logger.error(f"[AGE_LOG] Özel yaş modeli hatası: {str(e)}. InsightFace yaşı ({raw_insightface_age}) kullanılacak.")
                    estimated_age = raw_insightface_age
            else:
                 estimated_age = raw_insightface_age
                 logger.info(f"[AGE_LOG] Özel yaş modeli yok, InsightFace yaşı ({estimated_age:.1f}) kullanılacak.")

        if estimated_age is None: # Should not happen if raw_insightface_age was not None initially, but double-check
             logger.warning("[AGE_LOG] Belirlenen yaş None kaldı, varsayılan (25) kullanılacak.")
             estimated_age = 25
        
        logger.info(f"[AGE_LOG] CLIP için kullanılacak yaş: {estimated_age:.1f}")

        # CLIP için yüz bölgesini çıkar
        face_roi = None
        try:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            h_img, w_img = full_image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            if x2 > x1 and y2 > y1:
                 face_roi = full_image[y1:y2, x1:x2]
            else:
                 logger.warning(f"[AGE_LOG] estimate_age: Geçersiz bbox koordinatları nedeniyle face_roi çıkarılamadı: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        except Exception as e:
            logger.error(f"[AGE_LOG] face_roi çıkarılırken hata: {str(e)}")

        # CLIP modeli ile güven skoru hesapla
        if face_roi is None:
             logger.warning("[AGE_LOG] face_roi yok, CLIP atlanıyor, varsayılan güven (0.5) dönülüyor.")
             confidence_score = 0.5
        else:
             logger.info(f"[AGE_LOG] _calculate_confidence_with_clip çağrılıyor... Yaş: {estimated_age:.1f}")
             confidence_score = self._calculate_confidence_with_clip(face_roi, estimated_age)
             logger.info(f"[AGE_LOG] _calculate_confidence_with_clip döndü. Güven: {confidence_score:.4f}")

        logger.info(f"[AGE_LOG] estimate_age tamamlandı. Dönen Yaş: {estimated_age:.1f}, Dönen Güven: {confidence_score:.4f}")
        return estimated_age, confidence_score

    def _calculate_confidence_with_clip(self, face_image, estimated_age):
        logger.info(f"[AGE_LOG] _calculate_confidence_with_clip başladı. Gelen Yaş: {estimated_age:.1f}, Görüntü Shape: {face_image.shape}")
        if self.clip_model is None or face_image.size == 0:
            logger.warning("[AGE_LOG] CLIP modeli yok veya yüz görüntüsü geçersiz, varsayılan güven (0.5) dönülüyor.")
            return 0.5
        try:
            # Görüntüyü RGB'ye dönüştür ve PIL formatına çevir
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # CLIP için ön işleme
            preprocessed_image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.clip_device)
            
            # Yaş tahminini doğrulamak için daha genel yaş kategorisi prompt'ları
            age = int(round(estimated_age))
            prompts = []
            
            # Yaş kategorisi belirle
            if age < 3:
                prompts.append("a photo of a baby or infant")
                prompts.append("this person is clearly younger than 5 years old")
                prompts.append("not an adult or teenager")
            elif age < 13:
                prompts.append("a photo of a child")
                prompts.append("this person appears to be between 3 and 12 years old")
                prompts.append("not an adult or infant")
            elif age < 20:
                prompts.append("a photo of a teenager")
                prompts.append("this person appears to be between 13 and 19 years old")
                prompts.append("not a child or a middle-aged adult")
            elif age < 40:
                prompts.append("a photo of a young adult")
                prompts.append("this person appears to be in their twenties or thirties")
                prompts.append("not a child or a senior citizen")
            elif age < 65:
                prompts.append("a photo of a middle-aged adult")
                prompts.append("this person appears to be between 40 and 64 years old")
                prompts.append("not a teenager or an elderly person")
            else:
                prompts.append("a photo of a senior citizen or elderly person")
                prompts.append("this person appears to be 65 years old or older")
                prompts.append("not a young adult or child")
            
            # CLIP ile benzerlik hesapla
            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                text_inputs = clip.tokenize(prompts).to(self.clip_device)
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Benzerlik skorlarını al (logit scale * 100)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                # En yüksek benzerlik skorunu al (normalize edilmiş)
                confidence_score = similarity.max().item()
            
            # Çok düşük veya çok yüksek skorları sınırlayabiliriz (isteğe bağlı)
            confidence_score = max(0.1, min(0.9, confidence_score)) # Güveni 0.1-0.9 arasına sıkıştır
            
            logger.info(f"[AGE_LOG] CLIP Yaş Kategorisi Promptları: {prompts}")
            logger.info(f"[AGE_LOG] CLIP Benzerlik Skorları: {similarity.cpu().numpy()}")
            logger.info(f"[AGE_LOG] _calculate_confidence_with_clip tamamlandı. Hesaplanan Güven: {confidence_score:.4f}")
            return confidence_score
            
        except Exception as e:
            logger.error(f"[AGE_LOG] CLIP ile güven skoru hesaplanırken hata: {str(e)}")
            return 0.5 # Hata durumunda varsayılan güven

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

    def get_faces(self, image: np.ndarray):
        # This method is not provided in the original file or the code block
        # It's assumed to exist as it's called in the estimate_age method
        pass

# Kullanım örneği:
# estimator = InsightFaceAgeEstimator()
# img = cv2.imread('face.jpg')
# age = estimator.estimate_age(img)
# print('Tahmini yaş:', age)

# Bu fonksiyonu analysis_service.py tarafından import edilebilmesi için ekliyoruz.
def get_age_estimator():
    """InsightFaceAgeEstimator sınıfından bir örnek döndürür."""
    return InsightFaceAgeEstimator() 