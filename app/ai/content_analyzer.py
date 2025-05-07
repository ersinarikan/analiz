import os
import numpy as np
import cv2
import logging
from flask import current_app
import tensorflow as tf
from ultralytics import YOLO
import torch
import clip  # CLIP modelini import ediyoruz
from PIL import Image  # CLIP için PIL gerekiyor
import shutil
import time
import json
import math
from app.utils.serialization_utils import convert_numpy_types_to_python

logger = logging.getLogger(__name__)

# Singleton pattern için model repository
_models_cache = {}

class ContentAnalyzer:
    """İçerik analiz sınıfı, görüntülerdeki şiddet, yetişkin içeriği, vb. kategorileri tespit eder."""
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementasyonu - tek bir instance oluşturur"""
        if cls._instance is None:
            cls._instance = super(ContentAnalyzer, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        """
        ContentAnalyzer sınıfını başlatır ve gerekli modelleri yükler.
        Bu sınıf, şiddet, yetişkin içeriği, taciz, silah, madde kullanımı tespiti için kullanılır.
        """
        # Eğer zaten başlatıldıysa tekrar başlatma
        if self.initialized:
            return
            
        try:
            # Model klasörünü belirle
            model_folder = current_app.config.get('MODELS_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'storage', 'models'))
            
            # YOLOv8 modelini yükle - nesne tespiti için hala faydalı
            self.yolo_model = self._load_yolo_model(model_folder)
            
            # CLIP modelini yükle - daha büyük ve daha doğru olan versiyonu seçiyoruz
            self.clip_model, self.clip_preprocess = self._load_clip_model()
            
            # Kategori tanımlayıcıları - CLIP için kullanılacak metinler
            # Daha spesifik, detaylı ve ayırt edici prompt'lar
            self.category_texts = {
                "violence": [
                    "a photo showing a physical altercation or fight",
                    "image containing blood, visible injuries, or harm",
                    "people engaged in violent combat or physical aggression",
                    "scene depicting brutality, assault, or physical harm",
                    "explicit visual of someone being attacked or hurt",
                    "image NOT showing peaceful interaction or calm scene"
                ],
                "adult_content": [
                    "explicit sexual imagery not suitable for minors",
                    "photo displaying nudity or intimate body parts",
                    "people engaged in sexual acts or revealing poses",
                    "provocative content intended for adult audiences only",
                    "sexually suggestive situation with minimal clothing",
                    "image NOT showing family-friendly or workplace-appropriate content"
                ],
                "harassment": [
                    "photo showing someone being intimidated or threatened",
                    "visual evidence of bullying or emotional abuse",
                    "scene depicting unwanted aggressive interaction",
                    "image of person showing clear distress from others' actions",
                    "situation where someone is cornered or mobbed",
                    "image NOT showing consensual or positive social interaction"
                ],
                "weapon": [
                    "clearly visible gun, rifle, or firearm in threatening context",
                    "person holding or aiming a weapon at others",
                    "dangerous weapon like knife or sword displayed aggressively",
                    "military or combat weapons in active use",
                    "arsenal of weapons or ammunition visible in image",
                    "image NOT showing sports equipment or kitchen implements"
                ],
                "drug": [
                    "visible illegal substances or drugs",
                    "people consuming drugs or using drug paraphernalia",
                    "drug preparation materials like syringes, pipes, or rolling papers",
                    "pills, powders, or substances arranged for consumption",
                    "scene of intoxication or substance abuse",
                    "image NOT showing prescribed medication or legal substances"
                ],
                "safe": [
                    "appropriate content suitable for all ages and contexts",
                    "wholesome image showing positive interactions",
                    "educational, informative, or neutral content",
                    "family-friendly scene without concerning elements",
                    "image completely free from violence, adult content, or dangerous items",
                    "content that could safely be shown in a school or workplace"
                ]
            }
            
            # Kategori text tokenları önceden hesapla
            self.category_text_features = {}
            for category, prompts in self.category_texts.items():
                text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    # Tüm prompt'ların ortalamasını al
                    avg_text_features = text_features.mean(dim=0)
                    avg_text_features /= avg_text_features.norm()
                    self.category_text_features[category] = avg_text_features
            
            # Skor geçmişini yükle
            self.score_history = self._load_score_history()
            
            self.initialized = True
            logger.info("ContentAnalyzer - CLIP modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Content analyzer model yükleme hatası: {str(e)}")
            self.initialized = False
            raise
    
    def _load_clip_model(self):
        """CLIP modelini yükler, önbellek kontrolü yapar"""
        cache_key = "clip_model"
        
        # Önbellekte varsa direkt döndür
        if cache_key in _models_cache:
            logger.info("CLIP modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
            
        try:
            logger.info("CLIP modeli yükleniyor (ViT-L/14@336px)")  # En büyük ve en doğru model
            # Model ve önişleyici yükle (GPU varsa GPU'ya taşı)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                # Önce en büyük modeli yüklemeyi dene
                model, preprocess = clip.load("ViT-L/14@336px", device=device)
                logger.info("ViT-L/14@336px CLIP modeli başarıyla yüklendi")
            except Exception as e:
                # Yükleme başarısız olursa daha küçük modele geri dön
                logger.warning(f"Büyük CLIP modeli yüklenemedi, daha küçük model kullanılıyor: {str(e)}")
                model, preprocess = clip.load("ViT-B/32", device=device)
                logger.info("ViT-B/32 CLIP modeli başarıyla yüklendi")
            
            logger.info(f"CLIP modeli başarıyla yüklendi, çalışma ortamı: {device}")
            
            # Modeli önbelleğe ekle
            _models_cache[cache_key] = (model, preprocess)
            return model, preprocess
        except Exception as e:
            logger.error(f"CLIP modeli yüklenemedi: {str(e)}")
            raise
    
    def _load_yolo_model(self, model_folder):
        """YOLOv8 modelini yükler, cache kontrolü yapar"""
        cache_key = "yolov8"
        
        # Önbellekte varsa direkt döndür
        if cache_key in _models_cache:
            logger.info(f"YOLOv8 modeli önbellekten kullanılıyor")
            return _models_cache[cache_key]
            
        # Model klasörünü oluştur
        yolo_model_path = os.path.join(model_folder, 'detection', 'yolov8n.pt')
        if not os.path.exists(os.path.dirname(yolo_model_path)):
            os.makedirs(os.path.dirname(yolo_model_path), exist_ok=True)
        
        try:
            # Model zaten var mı kontrolü
            if os.path.exists(yolo_model_path):
                logger.info(f"Mevcut YOLOv8 modeli yükleniyor: {yolo_model_path}")
                model = YOLO(yolo_model_path)
            else:
                # Model yoksa indir ve kaydet
                logger.info(f"YOLOv8 modeli indiriliyor: {yolo_model_path}")
                model = YOLO('yolov8n.pt')
                
                # İndirilen modeli kopyala
                if os.path.exists('yolov8n.pt'):
                    shutil.copy('yolov8n.pt', yolo_model_path)
                    logger.info(f"YOLOv8 modeli başarıyla kopyalandı: {yolo_model_path}")
            
            logger.info(f"YOLOv8 modeli başarıyla yüklendi")
            
            # Modeli önbelleğe ekle
            _models_cache[cache_key] = model
            return model
        except Exception as yolo_err:
            logger.error(f"YOLOv8 modeli yüklenemedi: {str(yolo_err)}")
            # Yeniden indirilmeye çalışılır
            try:
                model = YOLO('yolov8n.pt')
                logger.info(f"YOLOv8 modeli online kaynaktan yüklendi")
                _models_cache[cache_key] = model
                return model
            except Exception as e:
                logger.error(f"YOLOv8 modeli online kaynaktan da yüklenemedi: {str(e)}")
                raise
    
    def analyze_image(self, image_path):
        """
        Bir resmi CLIP modeli ile analiz eder ve içerik skorlarını hesaplar.
        
        Args:
            image_path: Analiz edilecek resmin dosya yolu
            
        Returns:
            tuple: (şiddet skoru, yetişkin içerik skoru, taciz skoru, silah skoru, madde kullanımı skoru, güvenli skoru, tespit edilen nesneler)
        """
        try:
            # OpenCV ile görüntüyü yükle (YOLO için)
            if isinstance(image_path, str):
                cv_image = cv2.imread(image_path)
                if cv_image is None:
                    raise ValueError(f"Resim yüklenemedi: {image_path}")
                # CLIP için PIL formatına dönüştür
                if os.path.exists(image_path):
                    pil_image = Image.open(image_path).convert("RGB")
                else:
                    # OpenCV görüntüsünü PIL'e dönüştür
                    cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(cv_rgb)
            else:
                cv_image = image_path  # Zaten numpy array
                # OpenCV görüntüsünü PIL'e dönüştür
                cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_rgb)
            
            # YOLOv8 ile nesne tespiti (ek bilgi için)
            results = self.yolo_model(cv_image)
            
            # Nesne tespiti sonuçlarını işle
            detected_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Kutu koordinatları
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Yuvarlama ve Python int türüne dönüştürme
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    
                    # Güven skoru ve sınıf ID'si
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Etiket
                    label = self.yolo_model.names[cls_id]
                    
                    # Nesne bilgisini ekle
                    detected_objects.append({
                        'label': label,
                        'confidence': conf,
                        'box': [x1, y1, w, h]
                    })
            
            # CLIP ile görüntüyü analiz et
            preprocessed_image = self.clip_preprocess(pil_image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Görüntü özelliklerini CLIP ile çıkar
            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Her kategori için benzerlik skorunu hesapla
            raw_scores = {}
            confidences = {}  # CLIP güven skorları
            
            for category, text_features in self.category_text_features.items():
                # Görüntü-metin benzerliğini hesapla
                similarity = torch.cosine_similarity(image_features, text_features.unsqueeze(0))
                similarity_score = similarity.item()
                
                # logit scale ile sıcaklık ayarı (daha net skorlar için)
                temperature = 10.0  # Daha düşük bir sıcaklık değeri kullanıyoruz 
                scaled_score = torch.sigmoid(similarity * temperature).item()
                
                raw_scores[category] = scaled_score
                
                # Güven skorunu hesapla - benzerlik değerinin mutlak değeri bir tür güven göstergesidir
                confidences[category] = abs(similarity_score)
                
                logger.info(f"CLIP skorları - {category}: {scaled_score:.4f}, benzerlik: {similarity_score:.4f}")
            
            # YOLO ile tespit edilen nesneleri kullanarak içerik bağlamını zenginleştir
            object_labels = [obj['label'].lower() for obj in detected_objects]
            person_count = object_labels.count('person')
            
            # Nesne tespitine dayalı bağlamsal ayarlamalar
            self._apply_contextual_adjustments(raw_scores, object_labels, person_count)
            
            # Percentile-bazlı normalizasyon yap
            normalized_scores = self._percentile_normalize(raw_scores)
            
            # Skor geçmişini güncelle
            self._update_score_history(raw_scores)
            
            # Finalize confidence scores - log ile güven skorlarını göster
            for category, confidence in confidences.items():
                logger.info(f"CLIP güven skoru: {confidence:.4f} (category={category})")
            
            # Kategorilere göre skorları döndür
            violence_score = normalized_scores['violence']
            adult_content_score = normalized_scores['adult_content']
            harassment_score = normalized_scores['harassment']
            weapon_score = normalized_scores['weapon']
            drug_score = normalized_scores['drug']
            safe_score = normalized_scores['safe']
            
            # NumPy türlerini Python türlerine dönüştür
            safe_objects = convert_numpy_types_to_python(detected_objects)
            
            # Tüm değerleri Python standart türlerine dönüştürerek döndür
            return (float(violence_score), float(adult_content_score), float(harassment_score), 
                   float(weapon_score), float(drug_score), float(safe_score), safe_objects)
        except Exception as e:
            logger.error(f"CLIP görüntü analizi hatası: {str(e)}")
            # Hata durumunda alternatif analiz yöntemi
            return self._fallback_analysis(cv_image)
    
    def _apply_contextual_adjustments(self, scores, object_labels, person_count):
        """
        Nesne tespitine dayalı bağlamsal ayarlamalar yapar.
        Bu fonksiyon, CLIP sonuçlarını tespit edilen nesnelere göre ayarlar.
        """
        # Silahla ilgili nesneler
        weapon_objects = ['gun', 'knife', 'rifle', 'pistol', 'shotgun', 'weapon']
        if any(obj in object_labels for obj in weapon_objects):
            # Silah varsa silah ve şiddet skorlarını artır
            scores['weapon'] = min(scores['weapon'] * 1.5, 1.0)
            scores['violence'] = min(scores['violence'] * 1.3, 1.0)
            scores['safe'] = max(scores['safe'] * 0.5, 0.0)
            logger.info("Tespit edilen silah nesneleri, silah/şiddet skorları artırıldı")
        
        # Madde kullanımı ile ilgili nesneler
        drug_objects = ['bottle', 'wine glass', 'cup', 'cigarette', 'syringe']
        if any(obj in object_labels for obj in drug_objects):
            scores['drug'] = min(scores['drug'] * 1.2, 1.0)
            logger.info("Tespit edilen madde kullanımı göstergeleri, madde skoru artırıldı")
        
        # Birden fazla kişi varsa ilişkilendirme
        if person_count >= 2:
            # Kişiler arası etkileşim olasılığını artır
            scores['harassment'] = min(scores['harassment'] * (1.0 + (person_count * 0.1)), 1.0)
            scores['adult_content'] = min(scores['adult_content'] * (1.0 + (person_count * 0.05)), 1.0)
            logger.info(f"{person_count} kişi tespit edildi, kişilerarası etkileşim skorları ayarlandı")
    
    def _percentile_normalize(self, scores):
        """
        Skorları geçmiş analiz skorlarından oluşan bir veri setiyle karşılaştırarak
        yüzdelik dilim sonuçları elde eder.
        
        Args:
            scores: Mevcut analiz skorları
            
        Returns:
            dict: Yüzdelik dilim skorları (0-1 arası)
        """
        normalized_scores = {}
        
        # Her kategori için percentile hesapla
        for category, score in scores.items():
            if category in self.score_history and len(self.score_history[category]) > 10:
                # Bu kategori için geçmiş skorları sırala
                history = sorted(self.score_history[category])
                
                # Bu skorun yüzdelik dilimini hesapla
                percentile = sum(1 for h in history if h <= score) / len(history)
                normalized_scores[category] = percentile
                logger.info(f"Percentile normalizasyon - {category}: {score:.4f} -> {percentile:.4f} (geçmiş: {len(history)} örnek)")
            else:
                # Yeterli tarihsel veri yoksa sigmoid ile normalize et
                normalized_scores[category] = 1.0 / (1.0 + math.exp(-5 * (score - 0.5)))
                logger.info(f"Sigmoid normalizasyon - {category}: {score:.4f} -> {normalized_scores[category]:.4f} (yeterli geçmiş veri yok)")
        
        # Sonuçları toplam 1.0 olacak şekilde yeniden normalize et
        total = sum(normalized_scores.values())
        if total > 0:
            for category in normalized_scores:
                normalized_scores[category] = normalized_scores[category] / total
        
        return normalized_scores
    
    def _load_score_history(self):
        """Skor geçmişini yükler veya yoksa yeni oluşturur"""
        history_path = os.path.join(current_app.config.get('MODELS_FOLDER', ''), 'clip_score_history.json')
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    logger.info(f"Skor geçmişi yüklendi: {sum(len(scores) for scores in history.values())} toplam örnek")
                    return history
            except Exception as e:
                logger.error(f"Skor geçmişi yüklenirken hata: {str(e)}")
        
        # Varsayılan boş geçmiş
        logger.info("Yeni skor geçmişi oluşturuluyor")
        return {
            "violence": [], 
            "adult_content": [], 
            "harassment": [], 
            "weapon": [], 
            "drug": [], 
            "safe": []
        }
    
    def _update_score_history(self, scores):
        """Yeni skorları geçmiş verisine ekler"""
        # Geçmiş veriye yeni skorları ekle
        for category, score in scores.items():
            if category in self.score_history:
                self.score_history[category].append(score)
                # Listeyi MAX_HISTORY ile sınırla
                MAX_HISTORY = 1000
                if len(self.score_history[category]) > MAX_HISTORY:
                    self.score_history[category] = self.score_history[category][-MAX_HISTORY:]
        
        # Periyodik olarak kaydet (her 10 ekleme sonrası)
        if sum(1 for scores in self.score_history.values() for _ in scores) % 10 == 0:
            self._save_score_history()
    
    def _save_score_history(self):
        """Skor geçmişini dosyaya kaydeder"""
        history_path = os.path.join(current_app.config.get('MODELS_FOLDER', ''), 'clip_score_history.json')
        
        try:
            # Dizini oluştur
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            # Dosyaya kaydet
            with open(history_path, 'w') as f:
                json.dump(self.score_history, f)
                
            logger.info(f"Skor geçmişi kaydedildi: {sum(len(scores) for scores in self.score_history.values())} toplam örnek")
        except Exception as e:
            logger.error(f"Skor geçmişi kaydedilirken hata: {str(e)}")
            
    def _fallback_analysis(self, image):
        """
        CLIP modeli başarısız olursa klasik analiz yöntemlerine geri dön.
        """
        try:
            logger.warning("CLIP analizi başarısız oldu, geleneksel analize dönülüyor")
            
            # Görüntü özelliklerini analiz et
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            
            # Renk analizleri
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            hue_var = np.std(hsv[:, :, 0]) / 180.0
            
            # Kenar tespiti
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (image.shape[0] * image.shape[1] * 255)
            
            # Şiddet skoru
            violence_score = 0.1 + (edge_density * 0.5) + (contrast * 0.2)
            violence_score = min(max(violence_score, 0.1), 0.5)
            
            # Yetişkin içeriği
            has_skin_tones = self._detect_skin_tones(image)
            adult_score = 0.05 + (saturation * 0.3) + (0.2 if has_skin_tones else 0.0)
            adult_score = min(max(adult_score, 0.05), 0.3)
            
            # Taciz
            harassment_score = 0.1 + (hue_var * 0.2)
            harassment_score = min(max(harassment_score, 0.1), 0.3)
            
            # Silah
            weapon_score = 0.05 + (edge_density * 0.4) + (contrast * 0.1)
            weapon_score = min(max(weapon_score, 0.05), 0.3)
            
            # Madde kullanımı
            drug_score = 0.05 + (saturation * 0.2)
            drug_score = min(max(drug_score, 0.05), 0.25)
            
            # Güvenli
            safe_score = 1.0 - max(violence_score, adult_score, harassment_score, weapon_score, drug_score)
            safe_score = max(safe_score, 0.0)
            
            # Normalize et
            scores = {
                'violence': violence_score,
                'adult_content': adult_score,
                'harassment': harassment_score,
                'weapon': weapon_score,
                'drug': drug_score,
                'safe': safe_score
            }
            
            normalized_scores = self._normalize_scores(scores)
            
            # Boş nesne listesi
            empty_objects = []
            
            return (float(normalized_scores['violence']), 
                    float(normalized_scores['adult_content']),
                    float(normalized_scores['harassment']), 
                    float(normalized_scores['weapon']),
                    float(normalized_scores['drug']), 
                    float(normalized_scores['safe']),
                    empty_objects)
        except Exception as e:
            logger.error(f"Fallback analiz hatası: {str(e)}")
            # En son çare olarak varsayılan değerler
            return (0.1, 0.1, 0.1, 0.1, 0.1, 0.5, [])
    
    def _detect_skin_tones(self, image):
        """Görüntüde cilt tonları olup olmadığını tespit eder (basit yaklaşım)"""
        try:
            # HSV renk uzayına dönüştür
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Cilt tonu aralığı (HSV'de)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            
            # Cilt tonu maskesi oluştur
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Cilt pikseli oranı hesapla
            skin_ratio = np.sum(mask) / (image.shape[0] * image.shape[1] * 255)
            
            # Belirli bir eşiğin üzerinde cilt tonu varsa true döndür
            return skin_ratio > 0.15  # %15'den fazla cilt tonu içeriyorsa
        
        except Exception as e:
            logger.error(f"Cilt tonu tespiti hatası: {str(e)}")
            return False 
    
    def _normalize_scores(self, scores):
        """
        Kategorik skorları normalize eder, böylece toplam %100 olur.
        Daha ayırt edici sonuçlar için, skorları sıralar ve sıralamaya göre ağırlıklandırır.
        
        Args:
            scores: Kategori skorlarını içeren sözlük
            
        Returns:
            dict: Normalize edilmiş skorları içeren sözlük
        """
        try:
            # Tüm skorlar çok düşükse (0.1'den küçük)
            if all(score < 0.1 for score in scores.values()):
                # Güvenli kategorisine daha yüksek değer ver
                normalized_scores = {key: 0.05 for key in scores}
                normalized_scores['safe'] = 0.75
                return normalized_scores
            
            # Skorları büyükten küçüğe sırala
            sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # İlk üç kategoriye daha yüksek değerler ver (toplam 1.0 olacak şekilde)
            normalized_scores = {key: 0.05 for key in scores}  # Başlangıç değeri
            
            # En yüksek skora sahip kategori %50, ikinci %20, üçüncü %10 alır
            weights = [0.50, 0.20, 0.10]
            remaining_weight = 0.20  # Kalan %20'yi diğer kategorilere dağıt
            
            # En yüksek skorlu 3 kategoriye özel ağırlıklar ver
            for i, (category, _) in enumerate(sorted_categories[:min(3, len(sorted_categories))]):
                if i < len(weights):
                    normalized_scores[category] = weights[i]
            
            # Kalan kategoriler için toplam 'remaining_weight' dağıt
            remaining_categories = len(sorted_categories) - min(3, len(sorted_categories))
            if remaining_categories > 0:
                for i, (category, _) in enumerate(sorted_categories[3:]):
                    normalized_scores[category] = remaining_weight / remaining_categories
            
            # Log çıktısı
            logger.info(f"Normalize edilmiş toplam skor: {sum(normalized_scores.values()):.4f}")
            for category, score in normalized_scores.items():
                logger.info(f"  {category}: {score:.4f} (%{score * 100:.1f})")
            
            return normalized_scores
        except Exception as e:
            logger.error(f"Skor normalizasyonu hatası: {str(e)}")
            # Hata durumunda orijinal skorları döndür
            return scores 