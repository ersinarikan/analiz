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

def find_latest_age_model(base_dir):
    """
    Belirtilen dizin içinde en son model dosyasını bulur.
    Args:
        base_dir: Modellerin bulunduğu ana dizin
    Returns:
        Model dosyasının tam yolu veya None (hiç model bulunamazsa)
    """
    if not os.path.exists(base_dir):
        return None
        
    model_dir = os.path.join(base_dir, 'versions', 'v1')
    if not os.path.exists(model_dir):
        return None
        
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
        
    # Çoğunlukla epoch sayısına göre sıralama yapabilmek için, model adından epoch bilgisini çek
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)\.pth', filename)
        if match:
            return int(match.group(1))
        return 0
        
    # En yüksek epoch sayısına sahip model dosyasını bul
    latest_model = sorted(model_files, key=extract_epoch, reverse=True)[0]
    return os.path.join(model_dir, latest_model)

class InsightFaceAgeEstimator:
    def __init__(self, det_size=(640, 640)):
        # Model dosya yolunu ayarla
        model_path = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_x')
        logger.info(f"InsightFaceAgeEstimator başlatılıyor. Model dizini: {model_path}")
        logger.info("NOT: Buffalo_x adı kullanılmasına rağmen şu anda buffalo_sc modeli kullanılmaktadır (geçici çözüm)")
        
        # Model dosyalarının varlığını kontrol et
        if not os.path.exists(model_path):
            logger.error(f"Model dosyaları bulunamadı: {model_path}")
            raise FileNotFoundError(f"Model dosyaları bulunamadı: {model_path}")
        
        # Modeli yerel dosyadan yükle
        try:
            self.model = insightface.app.FaceAnalysis(
                name='buffalo_x',  # Klasör adı burada kullanılıyor
                root=os.path.dirname(os.path.dirname(model_path)),  # storage/models klasörü
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
            
            # Çocuk
            if age <= 12:
                category_prompts.extend([
                    "This is a child face",
                    "This person is a young child",
                    "This is a primary school aged kid"
                ])
                if age <= 5:
                    category_prompts.append("This is a toddler or very young child")
                    
            # Genç/Ergen
            if 10 <= age <= 19:
                category_prompts.extend([
                    "This is a teenage face",
                    "This is a young adolescent person",
                    "This face belongs to a teenager"
                ])
                
            # Genç Yetişkin
            if 18 <= age <= 30:
                category_prompts.extend([
                    "This is a young adult face",
                    "This person is in their twenties",
                    "This is a college-aged young adult"
                ])
                
            # Orta Yaşlı Yetişkin  
            if 30 <= age <= 50:
                category_prompts.extend([
                    "This is a middle-aged adult face",
                    "This person is in their thirties or forties",
                    "This face shows signs of early maturity"
                ])
                
            # Olgun Yetişkin
            if 50 <= age <= 65:
                category_prompts.extend([
                    "This is a mature adult face",
                    "This person is in their fifties or early sixties",
                    "This face shows signs of aging"
                ])
                
            # Yaşlı
            if age >= 65:
                category_prompts.extend([
                    "This is a senior citizen face",
                    "This person is elderly",
                    "This face shows significant signs of aging",
                    "This is an older person over 65"
                ])
                
            # 3. Negatif prompt'lar (farklı yaş grupları için)
            negative_prompts = []
            
            # Eğer çocuk değilse
            if age > 18:
                negative_prompts.extend([
                    "This is a child face",
                    "This person is a young child",
                    "This is a primary school aged kid"
                ])
                
            # Eğer genç/ergen değilse
            if age < 10 or age > 20:
                negative_prompts.extend([
                    "This is a teenage face",
                    "This is a young adolescent person",
                ])
                
            # Eğer yaşlı değilse
            if age < 60:
                negative_prompts.extend([
                    "This is a senior citizen face",
                    "This person is elderly",
                    "This face shows significant signs of aging"
                ])
                
            # Tüm prompt'ları birleştir
            all_prompts = age_prompts + category_prompts
            
            # Her prompt için benzerlik skoru hesapla
            text_inputs = clip.tokenize(all_prompts).to(self.clip_device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                text_features = self.clip_model.encode_text(text_inputs)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarity_values = similarity_scores.cpu().numpy()[0]
            
            # En büyük benzerlik skorlarını al (top-3)
            top_indices = np.argsort(similarity_values)[-3:][::-1]
            top_scores = similarity_values[top_indices]
            
            # Negatif prompt'larla kontrol
            negative_confidence = 1.0
            if negative_prompts:
                neg_text_inputs = clip.tokenize(negative_prompts).to(self.clip_device)
                with torch.no_grad():
                    neg_text_features = self.clip_model.encode_text(neg_text_inputs)
                    neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)
                    neg_similarity = (100.0 * image_features @ neg_text_features.T).softmax(dim=-1)
                    neg_values = neg_similarity.cpu().numpy()[0]
                
                # En yüksek negatif skorunu al ve güven skorundan çıkar
                max_neg_score = np.max(neg_values)
                negative_confidence = 1.0 - max_neg_score
            
            # Son güven skorunu hesapla
            weighted_score = np.mean(top_scores) * 0.7 + top_scores[0] * 0.3
            final_confidence = weighted_score * negative_confidence
            
            # Sigmoid fonksiyonu ile 0-1 aralığına normalize et
            normalized_confidence = 1.0 / (1.0 + math.exp(-10 * (final_confidence - 0.5)))
            
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