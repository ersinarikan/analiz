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
                providers=['CPUExecutionProvider'],
                det_thresh=0.2  # Yüz tespiti için minimum güven eşiği 0.3'ten 0.2'ye düşürüldü
            )
            self.model.prepare(ctx_id=0, det_size=det_size)
            logger.info("InsightFace temel modeli başarıyla yüklendi (det_thresh=0.2 ile)")
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
            logger.info("ViT-L/14@336px CLIP modeli yükleniyor - en büyük ve doğru model")
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=device)
            logger.info("ViT-L/14@336px CLIP modeli başarıyla yüklendi (yaş tahmini için)")
            
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
            
            # Yaş tahminini doğrulamak için özelleştirilmiş prompt'lar oluştur
            age = int(round(estimated_age))
            
            # 1. Direkt yaş aralığı prompt'ları (daha spesifik)
            age_decade = age // 10 * 10
            age_prompts = [
                f"This is a clear photo of a person who is exactly {age} years old",
                f"This face appears to be {age} years old",
                f"A person who is approximately {age-2}-{age+2} years old",
                f"This photo shows a typical face of someone in their {age_decade}s"
            ]
            
            # 2. Yaş kategorileri prompt'ları
            category_prompts = []
            if age < 3:
                category_prompts.append("This is a baby or infant (0-2 years old)")
            elif age < 10:
                category_prompts.append("This is a young child (3-9 years old)")
            elif age < 13:
                category_prompts.append("This is a pre-teen child (10-12 years old)")
            elif age < 20:
                category_prompts.append("This is a teenager (13-19 years old)")
            elif age < 30:
                category_prompts.append("This is a young adult in their twenties (20-29)")
            elif age < 40:
                category_prompts.append("This is an adult in their thirties (30-39)")
            elif age < 50:
                category_prompts.append("This is a middle-aged person in their forties (40-49)")
            elif age < 60:
                category_prompts.append("This is a middle-aged person in their fifties (50-59)")
            elif age < 70:
                category_prompts.append("This is a senior in their sixties (60-69)")
            else:
                category_prompts.append("This is an elderly person (70+ years old)")
                
            # 3. Karşıt prompt'lar (daha belirgin sonuçlar için)
            contrast_prompts = []
            if age < 18:
                contrast_prompts.append("This is an adult over 18 years old")
            else:
                contrast_prompts.append("This is a child under 18 years old")
                
            if age < 40:
                contrast_prompts.append("This is a middle-aged or elderly person (40+ years)")
            else:
                contrast_prompts.append("This is a young person under 40")
                
            # Tüm prompt'ları birleştir
            all_prompts = age_prompts + category_prompts + contrast_prompts
            logger.debug(f"[AGE_LOG] CLIP Prompts: {all_prompts}") # Debug seviyesinde logla
            
            # Prompt'ları tokenize et
            text_inputs = torch.cat([clip.tokenize(prompt) for prompt in all_prompts]).to(self.clip_device)
            
            # Görüntü ve metin özelliklerini çıkar
            with torch.no_grad():
                # Görüntü özelliklerini çıkar
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Metin özelliklerini çıkar
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Benzerlik skorlarını hesapla
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            logger.debug(f"[AGE_LOG] CLIP Ham Benzerlikler: {similarities.cpu().numpy().flatten()}") # Debug
            
            # Tüm pozitif prompt'ların ortalama benzerliğini al (karşıt prompt'lar hariç)
            positive_similarities = similarities[0, :len(age_prompts) + len(category_prompts)]
            avg_similarity = positive_similarities.mean().item()
            
            # Karşıt prompt'ların skoru düşükse bu iyi bir işaret (ters ölçekleme)
            contrast_similarities = similarities[0, len(age_prompts) + len(category_prompts):]
            inverted_contrast = 1.0 - contrast_similarities.mean().item()
            
            # Güven skorunu hesapla (pozitif ve negatif sinyalleri birleştir)
            confidence_score = (avg_similarity * 0.7) + (inverted_contrast * 0.3)
            
            # Sigmoid fonksiyonu ile 0-1 aralığına normalize et
            # İsteğe bağlı olarak sıcaklık parametresi ile keskinliği ayarla
            temperature = 2.0  # Daha yüksek = daha keskin ayrım
            normalized_confidence = 1.0 / (1.0 + math.exp(-temperature * (confidence_score - 0.5)))
            
            logger.info(f"[AGE_LOG] _calculate_confidence_with_clip tamamlandı. Hesaplanan Güven: {normalized_confidence:.4f}")
            return normalized_confidence
            
        except Exception as e:
            logger.error(f"[AGE_LOG] _calculate_confidence_with_clip Hata: {str(e)}")
            return 0.5

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