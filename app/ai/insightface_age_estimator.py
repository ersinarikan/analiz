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

    def estimate_age(self, image: np.ndarray):
        """
        Verilen görüntüdeki ilk yüzün yaş tahminini döndürür.
        Args:
            image: BGR (OpenCV) formatında numpy array
        Returns:
            Tuple: (age, confidence_score) veya (None, None)
        """
        faces = self.model.get(image)
        if not faces:
            logger.warning("Görüntüde yüz tespit edilemedi")
            # None yerine varsayılan değer döndür
            return 25.0, 0.5
        
        face = faces[0]
        # Yüz bölgesini çıkar
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        # Geçerlilik kontrolü
        if x1 < 0 or y1 < 0 or x2 >= image.shape[1] or y2 >= image.shape[0] or x2 <= x1 or y2 <= y1:
            logger.warning(f"Geçersiz yüz koordinatları: ({x1}, {y1}, {x2}, {y2}), görüntü boyutu: {image.shape}")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1] - 1, x2)
            y2 = min(image.shape[0] - 1, y2)
            
        face_image = image[y1:y2, x1:x2]
        
        # InsightFace ile yaş tahminini al
        if self.age_model is not None:
            try:
                embedding = face.embedding
                with torch.no_grad():
                    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                    age_pred = self.age_model(emb_tensor).item()
                logger.info(f"Özel model ile yaş tahmini yapıldı: {age_pred:.1f} yaş")
                estimated_age = age_pred
            except Exception as e:
                logger.error(f"Özel model ile yaş tahmini hatası: {str(e)}")
                logger.info("Varsayılan InsightFace yaş tahminine geçiliyor")
                estimated_age = face.age if face.age is not None else 25  # Varsayılan yaş
        else:
            estimated_age = face.age if face.age is not None else 25  # Varsayılan yaş
            logger.info(f"InsightFace ile yaş tahmini yapıldı: {estimated_age:.1f} yaş")
        
        # Yaş tahmini None ise varsayılan değer kullan
        if estimated_age is None:
            logger.warning("InsightFace yaş tahmini None döndürdü, varsayılan değer (25) kullanılıyor")
            estimated_age = 25  # Varsayılan yaş
            
        # CLIP modeli ile güven skoru hesapla
        logger.info(f"CLIP ile güven skoru hesaplanıyor... (yaş={estimated_age})")
        confidence_score = self._calculate_confidence_with_clip(face_image, estimated_age)
        
        logger.info(f"Yaş tahmini sonucu: {estimated_age:.1f} yaş, güven skoru: {confidence_score:.2f}")
        return estimated_age, confidence_score

    def _calculate_confidence_with_clip(self, face_image, estimated_age):
        """
        CLIP modeli kullanarak yaş tahmini için güven skoru hesaplar.
        Args:
            face_image: Yüz bölgesi görüntüsü (BGR)
            estimated_age: InsightFace modeli tarafından tahmin edilen yaş
        Returns:
            float: 0.0 ile 1.0 arasında güven skoru
        """
        if self.clip_model is None or face_image.size == 0:
            logger.warning("CLIP modeli kullanılamıyor veya yüz görüntüsü geçersiz, varsayılan güven skoru (0.5) döndürülüyor")
            return 0.5  # Varsayılan güven skoru
            
        try:
            # Görüntüyü RGB'ye dönüştür ve PIL formatına çevir
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # CLIP için ön işleme
            logger.debug("CLIP için görüntü ön işleme yapılıyor")
            preprocessed_image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.clip_device)
            
            # Yaş tahminini doğrulamak için özelleştirilmiş prompt'lar oluştur
            age = int(round(estimated_age))
            
            # 1. Direkt yaş aralığı prompt'ları (daha spesifik)
            age_decade = age // 10 * 10
            age_prompts = [
                f"This is a clear photo of a person who is exactly {age} years old",
                f"This face appears to be {age} years old",
                f"A person who is approximately {age-2}-{age+2} years old",
                f"This photo shows a typical face of someone in their {age_decade}s",
                f"The facial features suggest a {age} year old person",
                f"Based on skin texture and facial structure, this person is {age} years old",
                f"The face in this image has characteristics of a {age}-year-old individual"
            ]
            
            # 2. Yaş ile ilgili fiziksel özellikler için ek promptlar
            physical_feature_prompts = [
                f"This face shows typical skin texture for a {age} year old",
                f"The facial features and proportions match a {age} year old",
                f"This face shows typical age markers for someone {age} years old",
                f"The aging signs in this face are consistent with a {age}-year-old person"
            ]
            
            # 3. Yaş kategorileri prompt'ları (daha detaylı ve ayrıntılı)
            category_prompts = []
            if age < 3:
                category_prompts.extend([
                    "This is a baby or infant (0-2 years old)",
                    "This is the face of an infant with baby features"
                ])
            elif age < 7:
                category_prompts.extend([
                    "This is a young child (3-6 years old)",
                    "This is a preschool or kindergarten age child"
                ])
            elif age < 13:
                category_prompts.extend([
                    "This is a pre-teen child (7-12 years old)",
                    "This is an elementary school age child"
                ])
            elif age < 18:
                category_prompts.extend([
                    "This is a teenager (13-17 years old)",
                    "This is an adolescent face with teenage features"
                ])
            elif age < 25:
                category_prompts.extend([
                    "This is a young adult (18-24 years old)",
                    "This is a college-age young adult"
                ])
            elif age < 35:
                category_prompts.extend([
                    "This is an adult in their late twenties or early thirties",
                    "This is a young professional adult (25-34)"
                ])
            elif age < 45:
                category_prompts.extend([
                    "This is an adult in their late thirties or early forties",
                    "This is a mid-career adult (35-44)"
                ])
            elif age < 55:
                category_prompts.extend([
                    "This is a middle-aged person in their late forties or early fifties",
                    "This is an adult with early signs of aging (45-54)"
                ])
            elif age < 65:
                category_prompts.extend([
                    "This is a person in their late fifties or early sixties",
                    "This is an older adult approaching retirement age"
                ])
            elif age < 75:
                category_prompts.extend([
                    "This is a senior in their late sixties or early seventies",
                    "This is a retirement-age senior adult"
                ])
            else:
                category_prompts.extend([
                    "This is an elderly person (75+ years old)",
                    "This is an older senior with advanced age features"
                ])
                
            # 4. Karşıt prompt'lar (daha belirgin sonuçlar için)
            contrast_prompts = []
            if age < 18:
                contrast_prompts.extend([
                    "This is an adult over 18 years old",
                    "This face has mature adult features"
                ])
            else:
                contrast_prompts.extend([
                    "This is a child under 18 years old",
                    "This face has juvenile features"
                ])
                
            if age < 30:
                contrast_prompts.append("This is a middle-aged or older person (over 45)")
            elif age < 60:
                contrast_prompts.append("This is either a very young person (under 25) or very old person (over 70)")
            else:
                contrast_prompts.append("This is a young person under 40")
                
            # Tüm prompt'ları birleştir
            all_prompts = age_prompts + physical_feature_prompts + category_prompts + contrast_prompts
            logger.debug(f"CLIP için kullanılan prompt'lar: {all_prompts}")
            
            # Prompt'ları tokenize et
            text_inputs = torch.cat([clip.tokenize(prompt) for prompt in all_prompts]).to(self.clip_device)
            
            # Görüntü ve metin özelliklerini çıkar
            with torch.no_grad():
                # Görüntü özelliklerini çıkar
                logger.debug("CLIP görüntü özellikleri çıkarılıyor")
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Metin özelliklerini çıkar
                logger.debug("CLIP metin özellikleri çıkarılıyor")
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Benzerlik skorlarını hesapla
            logger.debug("CLIP benzerlik skorları hesaplanıyor")
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
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
            
            logger.info(f"CLIP ile yaş tahmini güven skoru: {normalized_confidence:.4f} (raw: {confidence_score:.4f}, avg_sim: {avg_similarity:.4f}, inv_contrast: {inverted_contrast:.4f})")
            
            return normalized_confidence
            
        except Exception as e:
            logger.error(f"CLIP güven skoru hesaplama hatası: {str(e)}")
            logger.warning("Güven skoru hatası, varsayılan değer (0.5) döndürülüyor")
            return 0.5  # Hata durumunda varsayılan değer

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