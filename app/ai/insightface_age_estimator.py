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
import traceback

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
            return None, 0.15  # En düşük güven skoruyla None döndür
        
        face = faces[0]
        # DEBUG: InsightFace'in döndürdüğü ham yüz özelliklerini logla
        logger.info(f"DEBUG - InightFace Ham Değerler: Yüz #0, Yaş={face.age}, InsightFace Güven Skoru={face.confidence}")
        
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
                estimated_age = face.age  # InsightFace yaşını kullan
        else:
            estimated_age = face.age  # InsightFace yaşını kullan
            logger.info(f"InsightFace ile yaş tahmini yapıldı: {estimated_age:.1f} yaş")
        
        # Yaş tahmini None ise varsayılan değer kullan
        if estimated_age is None:
            logger.warning("InsightFace yaş tahmini None döndürdü, varsayılan değer (25) kullanılıyor")
            estimated_age = 25  # Varsayılan yaş
            
        # CLIP modeli ile güven skoru hesapla
        logger.info(f"CLIP ile güven skoru hesaplanıyor... (yaş={estimated_age})")
        confidence_score = self._calculate_confidence_with_clip(face_image, estimated_age)
        
        # NOT: InsightFace'in güven skorunu tamamen yok sayıyor ve sadece CLIP'in güven skorunu kullanıyoruz
        logger.info(f"Yaş tahmini sonucu: {estimated_age:.1f} yaş, CLIP güven skoru: {confidence_score:.2f}")
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
        # CLIP veya yüz görüntüsü eksikse düşük varsayılan değer döndür
        # Değişiklik: 0.5 yerine 0.15 kullanıyoruz (kötü kareler için düşük skor)
        if self.clip_model is None or face_image.size == 0:
            logger.warning("CLIP modeli kullanılamıyor veya yüz görüntüsü geçersiz, düşük güven skoru (0.15) döndürülüyor")
            return 0.15  # Düşük varsayılan güven skoru
            
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
                f"The face in this image has characteristics of a {age}-year-old individual",
                f"This person appears to be around {age} years of age",
                f"This is a typical {age} year old face with appropriate features",
                f"The age of this person is visibly {age} years based on facial features"
            ]
            
            # 2. Yaş ile ilgili fiziksel özellikler için ek promptlar
            physical_feature_prompts = [
                f"This face shows typical skin texture for a {age} year old",
                f"The facial features and proportions match a {age} year old",
                f"This face shows typical age markers for someone {age} years old",
                f"The aging signs in this face are consistent with a {age}-year-old person",
                f"The complexion and facial structure indicate a {age} year old person",
                f"The wrinkles and facial lines are typical for a {age} year old person",
                f"The skin elasticity appears to match that of a {age} year old",
                f"Facial bone structure suggests a person of {age} years"
            ]
            
            # YENİ: Daha net ve basit tanımlamalar
            new_simple_prompts = [
                f"This person is {age} years old",
                f"This face is between {age-5} and {age+5} years old",
                f"The person in this photo appears to be {age} years old",
                f"This face matches a {age} year old person",
                f"This person is in their {age_decade}s"
            ]

            # YENİ: Görsel özelliklere odaklanan promptlar
            new_visual_prompts = [
                f"The wrinkle level in this face matches {age} years",
                f"The skin texture shows typical features of {age} years",
                f"The face contours are typical of a {age} year old",
                "This is a face photograph",
                "This image shows a person's face"
            ]

            # YENİ: Yüksek güven skorları için ek genel promptlar
            general_face_prompts = [
                "This is a photograph of a person",
                "This image contains a human face",
                "This is a portrait photo",
                "There is a person in this picture",
                "This is a clear image of a face"
            ]
            
            # 3. Yaş kategorileri prompt'ları (daha detaylı ve ayrıntılı)
            category_prompts = []
            if age < 3:
                category_prompts.extend([
                    "This is a baby or infant (0-2 years old)",
                    "This is the face of an infant with baby features",
                    "This is an infant with typical baby facial characteristics",
                    "The rounded cheeks and facial structure indicate a baby",
                    "This is a very young child with typical infant features",
                    "The facial proportions are characteristic of a baby under 3 years old"
                ])
            elif age < 7:
                category_prompts.extend([
                    "This is a young child (3-6 years old)",
                    "This is a preschool or kindergarten age child",
                    "The facial features indicate a young child between 3-6 years",
                    "This is a child of early elementary school age",
                    "The face has proportions typical of a 3-6 year old child",
                    "This is a young child with characteristic preschool-age features"
                ])
            elif age < 13:
                category_prompts.extend([
                    "This is a pre-teen child (7-12 years old)",
                    "This is an elementary school age child",
                    "The facial development indicates a school-age child",
                    "This face shows characteristics of a pre-teen child",
                    "The facial structure is typical of a child aged 7-12 years",
                    "This is a child approaching adolescence"
                ])
            elif age < 18:
                category_prompts.extend([
                    "This is a teenager (13-17 years old)",
                    "This is an adolescent face with teenage features",
                    "This person is clearly in the middle of adolescent development",
                    "The facial features show typical teenage characteristics",
                    "This is a young person in adolescence with characteristic features",
                    "This face shows the developmental features of a 13-17 year old"
                ])
            elif age < 25:
                category_prompts.extend([
                    "This is a young adult (18-24 years old)",
                    "This is a college-age young adult",
                    "The facial features indicate an early 20s adult",
                    "This is a young adult with fully developed facial features",
                    "The face shows the characteristics of a young adult aged 18-24",
                    "This person appears to be a young adult in their early twenties"
                ])
            elif age < 35:
                category_prompts.extend([
                    "This is an adult in their late twenties or early thirties",
                    "This is a young professional adult (25-34)",
                    "The facial features indicate a person in their early thirties",
                    "This is an adult with fully mature facial features in their late 20s",
                    "The face shows characteristics of an adult in their early 30s",
                    "This is a person in the prime of early adulthood"
                ])
            elif age < 45:
                category_prompts.extend([
                    "This is an adult in their late thirties or early forties",
                    "This is a mid-career adult (35-44)",
                    "The face shows early signs of aging consistent with early 40s",
                    "This is a mature adult with well-defined facial features",
                    "The facial structure is typical of someone in their late 30s",
                    "This person shows the balanced maturity of middle adulthood"
                ])
            elif age < 55:
                category_prompts.extend([
                    "This is a middle-aged person in their late forties or early fifties",
                    "This is an adult with early signs of aging (45-54)",
                    "The face shows defined wrinkles typical of early 50s",
                    "This person has the facial characteristics of middle age",
                    "The signs of aging in this face are consistent with someone in their late 40s",
                    "This is a mature adult with visible aging markers in their early 50s"
                ])
            elif age < 65:
                category_prompts.extend([
                    "This is a person in their late fifties or early sixties",
                    "This is an older adult approaching retirement age",
                    "The face shows pronounced aging signs of someone in their early 60s",
                    "This person has the facial features typical of the early senior years",
                    "The skin texture and wrinkles indicate a person around 60 years old",
                    "This face shows the distinguished aging of late middle age"
                ])
            elif age < 75:
                category_prompts.extend([
                    "This is a senior in their late sixties or early seventies",
                    "This is a retirement-age senior adult",
                    "The face shows significant aging consistent with early 70s",
                    "This person has the facial characteristics of a senior citizen",
                    "The facial appearance matches that of someone in their late 60s",
                    "This is an elderly person with well-defined aging features"
                ])
            else:
                category_prompts.extend([
                    "This is an elderly person (75+ years old)",
                    "This is an older senior with advanced age features",
                    "The face shows significant aging markers of someone over 75",
                    "This person has the deep wrinkles and features of advanced age",
                    "The facial structure is typical of someone in their late 70s or older",
                    "This is an elderly person with prominent signs of advanced aging"
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
                
            # Tüm prompt'ları birleştir - YENİ promptlar dahil
            all_prompts = age_prompts + physical_feature_prompts + category_prompts + new_simple_prompts + new_visual_prompts + general_face_prompts + contrast_prompts
            
            # DEBUG: Toplam prompt sayısını logla
            logger.info(f"DEBUG - Toplam prompt sayısı: {len(all_prompts)}")
            
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
            
            # DEBUG: Her promptun skorunu logla
            similarities_list = similarities[0].cpu().tolist()
            for i, (prompt, score) in enumerate(zip(all_prompts, similarities_list)):
                if i < 5:  # İlk 5 promptu tam olarak logla
                    logger.info(f"DEBUG - Prompt #{i+1}: '{prompt}' - Skor: {score:.4f}")
                
            # DEBUG: En yüksek ve en düşük skorlu promptları göster
            top_indices = torch.topk(similarities[0], 3).indices.cpu().tolist()
            bottom_indices = torch.topk(similarities[0], 3, largest=False).indices.cpu().tolist()
            
            logger.info("DEBUG - En yüksek skorlu promptlar:")
            for idx in top_indices:
                logger.info(f"  '{all_prompts[idx]}' - Skor: {similarities_list[idx]:.4f}")
                
            logger.info("DEBUG - En düşük skorlu promptlar:")
            for idx in bottom_indices:
                logger.info(f"  '{all_prompts[idx]}' - Skor: {similarities_list[idx]:.4f}")
            
            # Pozitif promptların indeksleri (kontrast promptlar hariç)
            pos_start = 0
            pos_end = len(all_prompts) - len(contrast_prompts)
            
            # *** YENİ: None değerleri güvenli şekilde işle ***
            try:
                # Pozitif promptlardan None olmayanları al
                positive_similarities = [s.item() for s in similarities[0, pos_start:pos_end] if not torch.isnan(s)]
                
                # Eğer geçerli pozitif benzerlik yoksa, varsayılan değer kullan
                if not positive_similarities:
                    logger.warning("Geçerli pozitif benzerlik yok, varsayılan güven skoru kullanılıyor")
                    return 0.5
                    
                # Pozitif promptların ortalaması
                avg_similarity = sum(positive_similarities) / len(positive_similarities)
                
                # Karşıt promptlardan None olmayanları al
                contrast_similarities = [s.item() for s in similarities[0, pos_end:] if not torch.isnan(s)]
                
                # Eğer geçerli karşıt benzerlik yoksa, sadece pozitif skoru kullan
                if not contrast_similarities:
                    logger.warning("Geçerli karşıt benzerlik yok, sadece pozitif skorlar kullanılıyor")
                    inverted_contrast = 0.5  # Nötr değer
                else:
                    # Karşıt promptların ortalamasını ters çevir
                    inverted_contrast = 1.0 - (sum(contrast_similarities) / len(contrast_similarities))
                
                # Güven skorunu hesapla
                confidence_score = (avg_similarity * 0.7) + (inverted_contrast * 0.3)
                
                # DEBUG: Skor detaylarını logla
                logger.info(f"DEBUG - CLIP Skor Detayları:")
                logger.info(f"  - Pozitif promptlar ortalaması: {avg_similarity:.4f} (toplam {len(positive_similarities)} geçerli prompt)")
                logger.info(f"  - Karşıt promptlar ters değeri: {inverted_contrast:.4f} (toplam {len(contrast_similarities)} geçerli prompt)")
                logger.info(f"  - Ağırlıklı final skor: {confidence_score:.4f}")
                
            except Exception as e:
                logger.error(f"Güven skoru hesaplama hatası: {str(e)}")
                logger.error(f"Hata ayrıntıları: {traceback.format_exc()}")
                return 0.5  # Herhangi bir hata olursa varsayılan değer
            
            # Sigmoid fonksiyonu ile 0-1 aralığına normalize et
            # İsteğe bağlı olarak sıcaklık parametresi ile keskinliği ayarla
            temperature = 2.0  # Daha yüksek = daha keskin ayrım
            base_confidence = 1.0 / (1.0 + math.exp(-temperature * (confidence_score - 0.5)))
            
            # YENİ: Güven skorunu yeniden ölçeklendir - düşük taban değeri + CLIP skoru
            # Bu yaklaşım, CLIP'in bir değer döndürdüğü tüm durumların None döndürülen 
            # durumlardan (0.15) daha yüksek skor almasını sağlar
            BASE_VALUE = 0.15  # CLIP None döndürdüğünde kullanılan değer
            MAX_VALUE = 0.95   # İzin verilen maksimum değer
            
            # ÖNEMLİ: Taban değerin üzerine ekle ve maksimum değerle sınırla
            normalized_confidence = BASE_VALUE + ((MAX_VALUE - BASE_VALUE) * base_confidence)
            normalized_confidence = min(normalized_confidence, MAX_VALUE)  # En fazla 0.95 olabilir
            
            logger.info(f"CLIP ile yaş tahmini güven skoru: {normalized_confidence:.4f} (raw: {confidence_score:.4f}, base: {base_confidence:.4f}, avg_sim: {avg_similarity:.4f}, inv_contrast: {inverted_contrast:.4f})")
            
            return normalized_confidence
            
        except Exception as e:
            logger.error(f"CLIP güven skoru hesaplama hatası: {str(e)}")
            logger.error(f"Hata ayrıntıları: {traceback.format_exc()}")
            logger.warning("Güven skoru hatası, varsayılan değer (0.5) döndürülüyor")
            return 0.5  # Hata durumunda orta-düzey varsayılan değer

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