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
            # Daha spesifik, detaylı ve ayırt edici prompt\'lar
            self.category_texts = {
                "violence": [
                    "graphic violence, fighting, or assault",
                    "images depicting severe injuries, blood, or gore",
                    "people physically attacking others",
                    "scenes of combat or warfare",
                    "a person visibly in pain or suffering due to aggression"
                ],
                "adult_content": [
                    "explicit nudity or sexually suggestive content",
                    "depiction of sexual acts or genitalia",
                    "content intended to cause sexual arousal",
                    "pornographic material",
                    "highly revealing clothing in a sexual context"
                ],
                "harassment": [
                    "images showing bullying, intimidation, or verbal abuse",
                    "a person looking distressed or fearful due to others\' actions",
                    "scenes of mobbing or targeted humiliation",
                    "content that demeans or attacks a specific group",
                    "non-consensual pursuit or stalking"
                ],
                "weapon": [
                    "a person holding or aiming a firearm (gun, rifle)",
                    "visible knives, swords, or daggers used threateningly",
                    "explosives or bombs",
                    "military weapons outside of a ceremonial context",
                    "close-up image of a dangerous weapon"
                ],
                "drug": [
                    "images depicting illegal drug use (injection, smoking, snorting)",
                    "visible illicit drugs like cocaine, heroin, or pills",
                    "drug paraphernalia such as syringes, pipes, or bongs",
                    "scenes suggesting drug dealing or manufacturing",
                    "person showing signs of drug overdose or severe intoxication"
                ],
                "safe": [
                    "a harmless scene with no offensive content",
                    "neutral images of objects, landscapes, or animals",
                    "people in normal, everyday situations",
                    "standard workplace or educational settings",
                    "family-friendly content suitable for all audiences"
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
        yolo_model_path = os.path.join(model_folder, 'detection', 'yolov8x.pt')
        if not os.path.exists(os.path.dirname(yolo_model_path)):
            os.makedirs(os.path.dirname(yolo_model_path), exist_ok=True)
        
        try:
            # Model zaten var mı kontrolü
            if os.path.exists(yolo_model_path):
                logger.info(f"Mevcut YOLOv8x modeli yükleniyor: {yolo_model_path}")
                model = YOLO(yolo_model_path)
            else:
                # Model yoksa indir ve kaydet
                logger.info(f"YOLOv8x modeli indiriliyor: {yolo_model_path}")
                model = YOLO('yolov8x.pt')
                
                # İndirilen modeli kopyala
                if os.path.exists('yolov8x.pt'):
                    shutil.copy('yolov8x.pt', yolo_model_path)
                    logger.info(f"YOLOv8x modeli başarıyla kopyalandı: {yolo_model_path}")
            
            logger.info(f"YOLOv8x modeli başarıyla yüklendi")
            
            # Modeli önbelleğe ekle
            _models_cache[cache_key] = model
            return model
        except Exception as yolo_err:
            logger.error(f"YOLOv8x modeli yüklenemedi: {str(yolo_err)}")
            # Yeniden indirilmeye çalışılır
            try:
                logger.info("YOLOv8x yüklenemedi, fallback olarak YOLOv8n deneniyor...")
                model = YOLO('yolov8n.pt')
                logger.info(f"YOLOv8n modeli online kaynaktan fallback olarak yüklendi")
                _models_cache[cache_key] = model
                return model
            except Exception as e:
                logger.error(f"Fallback YOLOv8n modeli de online kaynaktan yüklenemedi: {str(e)}")
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
                temperature = 1.0 
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
            adjusted_scores = raw_scores # Artık percentile normalizasyon olmadığı için ayarlanmış skorlar doğrudan bunlar olacak.

            # Skor geçmişini güncelle
            self._update_score_history(raw_scores)
            
            # Finalize confidence scores - log ile güven skorlarını göster
            for category, confidence in confidences.items():
                logger.info(f"CLIP güven skoru: {confidence:.4f} (category={category})")
            
            # Kategorilere göre skorları döndür
            violence_score = adjusted_scores['violence']
            adult_content_score = adjusted_scores['adult_content']
            harassment_score = adjusted_scores['harassment']
            weapon_score = adjusted_scores['weapon']
            drug_score = adjusted_scores['drug']
            safe_score = adjusted_scores['safe']
            
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
        # Silahla ilgili nesneler ve potansiyel riskli nesneler
        weapon_objects = ['gun', 'rifle', 'pistol', 'shotgun', 'weapon', 'explosive', 'bomb']
        drug_objects = ['bottle', 'wine glass', 'cup', 'cigarette', 'syringe', 'pipe', 'bong', 'pills', 'powder'] # 'bottle', 'cup' gibi genel nesneler dikkatli kullanılmalı
        
        # Mutfakla ilgili nesneler (yanlış pozitifleri azaltmak için)
        kitchen_objects = ['oven', 'refrigerator', 'sink', 'microwave', 'kitchen', 'restaurant', 'dining table', 'food', 'plate']

        # Riskli nesne tespiti yapıldı mı?
        weapon_detected = any(obj in object_labels for obj in weapon_objects)
        drug_related_detected = any(obj in object_labels for obj in drug_objects if obj not in ['bottle', 'cup']) # Daha spesifik uyuşturucu nesneleri
        general_drug_indicators = any(obj in ['bottle', 'cup'] for obj in object_labels) # Şişe, bardak gibi genel ama bağlama göre riskli olabilecekler

        if weapon_detected:
            is_kitchen_context_with_knife = 'knife' in object_labels and any(ko in object_labels for ko in kitchen_objects)
            if is_kitchen_context_with_knife:
                # Mutfak bıçağı daha az riskli kabul edilebilir
                scores['weapon'] = min(scores['weapon'] * 1.1, 1.0) # Hafif artış
                scores['violence'] = min(scores['violence'] * 1.1, 1.0)
                scores['safe'] = max(scores['safe'] * 0.8, 0.0) 
                logger.info("Mutfak bağlamında bıçak tespit edildi, silah/şiddet skorları daha az artırıldı.")
            else:
                # Genel silah tespiti
                scores['weapon'] = min(scores['weapon'] * 1.5, 1.0)
                scores['violence'] = min(scores['violence'] * 1.3, 1.0)
                scores['safe'] = max(scores['safe'] * 0.5, 0.0)
                logger.info("Tespit edilen silah nesneleri, silah/şiddet skorları artırıldı")
        
        if drug_related_detected:
            scores['drug'] = min(scores['drug'] * 1.4, 1.0) # Spesifik uyuşturucu nesneleri için daha güçlü artış
            scores['safe'] = max(scores['safe'] * 0.6, 0.0)
            logger.info("Tespit edilen spesifik madde kullanımı göstergeleri, madde skoru artırıldı")
        elif general_drug_indicators and person_count > 0 : # Şişe, bardak gibi nesneler ve insanlar varsa
            # Bu durum daha belirsiz olduğu için 'drug' skorunu daha az etkileyebilir veya ek bağlam gerekebilir
            # Örneğin, 'party' (parti) gibi bir nesne tespitiyle birleştirilebilir
            if 'party' in object_labels or 'bar' in object_labels:
                 scores['drug'] = min(scores['drug'] * 1.1, 1.0)
                 logger.info("Genel madde kullanımı göstergeleri (şişe/bardak) parti/bar bağlamında, madde skoru hafif artırıldı")

        # Birden fazla kişi varsa kişilerarası etkileşim olasılığı
        if person_count >= 2:
            scores['harassment'] = min(scores['harassment'] * (1.0 + (person_count * 0.08)), 1.0) # Çarpanı biraz düşürdük
            scores['adult_content'] = min(scores['adult_content'] * (1.0 + (person_count * 0.04)), 1.0) # Çarpanı biraz düşürdük
            if person_count > 5: # Çok kalabalık durumlar için
                 scores['safe'] = max(scores['safe'] * 0.9, 0.0) # Kalabalıklar bazen daha az güvenli olabilir
            logger.info(f"{person_count} kişi tespit edildi, kişilerarası etkileşim skorları ayarlandı")
        
        # GÜVENLİ KATEGORİSİ İÇİN GÜÇLENDİRME
        # Eğer belirgin bir riskli nesne yoksa ve kişi sayısı azsa 'safe' skorunu artır
        no_immediate_risk_objects = not weapon_detected and not drug_related_detected
        
        # Diğer kategorilerin (safe hariç) skorları genel olarak düşük mü kontrolü
        # Bu kontrol, _apply_contextual_adjustments fonksiyonuna gelen orijinal 'scores' (scaled_scores) üzerinden yapılmalı
        # Henüz 'safe' skoru bu blok içinde yapay olarak artırılmadı veya diğerleri düşürülmedi.
        other_categories_for_safe_check = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
        all_other_scores_truly_low = all(scores.get(cat, 0) < 0.3 for cat in other_categories_for_safe_check)
        # Eşik değeri (0.3) ayarlanabilir bir parametre olabilir.

        if no_immediate_risk_objects and all_other_scores_truly_low and person_count <= 1:
            # Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve ortam sakin görünüyorsa
            original_safe_score = scores.get('safe', 0)
            scores['safe'] = min(original_safe_score * 1.5, 1.0)  # Safe skorunu belirgin şekilde artır (üst sınır 1.0)
            
            reduction_factor = 0.6 # Diğer skorlar için daha güçlü bir azaltma faktörü
            if original_safe_score > 0.7: # Eğer orijinal safe skoru zaten yüksekse, diğerlerini daha da fazla düşür
                reduction_factor = 0.4
            
            for category in other_categories_for_safe_check:
                scores[category] = max(scores.get(category, 0) * reduction_factor, 0.0)
            logger.info(f"Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve az kişi var, 'safe' skoru güçlendirildi ({original_safe_score:.2f} -> {scores['safe']:.2f}), diğerleri düşürüldü.")
        elif no_immediate_risk_objects and all_other_scores_truly_low and person_count > 1 and person_count <=3:
            # Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve 2-3 kişi var, safe yine de baskın olabilir
            original_safe_score = scores.get('safe', 0)
            scores['safe'] = min(original_safe_score * 1.3, 1.0) # Safe skorunu artır
            reduction_factor = 0.7
            if original_safe_score > 0.6:
                 reduction_factor = 0.5
            for category in other_categories_for_safe_check:
                scores[category] = max(scores.get(category, 0) * reduction_factor, 0.0)
            logger.info(f"Belirgin YOLO riski yok, TÜM DİĞER CLIP SKORLARI DÜŞÜK ve 2-3 kişi var, 'safe' skoru artırıldı ({original_safe_score:.2f} -> {scores['safe']:.2f}), diğerleri düşürüldü.")

        # CLIP SKORU YÜKSEK AMA YOLO ONAYI YOKSA DÜŞÜRME MANTIĞI
        # Bu blok, yukarıdaki 'safe' ayarlamalarından sonra çalışmalı ki,
        # 'safe' zaten diğer skorları düşürmüşse tekrar aşırı düşürmesin, ya da tam tersi.
        # Ancak mevcut durumda 'safe' ayarlaması sadece 'safe' olmayan kategorileri etkiliyor.
        # Bu düşürme, 'safe' dışındaki kategorilere odaklanmalı.

        high_clip_threshold = 0.7 # CLIP skorunun yüksek kabul edileceği eşik
        yolo_miss_reduction_factor = 0.5 # YOLO onayı olmadığında uygulanacak azaltma faktörü

        # Silah için
        if scores.get('weapon', 0) > high_clip_threshold and not weapon_detected:
            original_weapon_score = scores['weapon']
            scores['weapon'] *= yolo_miss_reduction_factor
            logger.info(f"CLIP 'weapon' skoru yüksek ({original_weapon_score:.2f}) ama YOLO onayı yok, skor {scores['weapon']:.2f}'ye düşürüldü.")

        # Madde için
        # 'general_drug_indicators' burada kafa karıştırabilir, çünkü bunlar zaten zayıf göstergeler.
        # Bu yüzden sadece 'drug_related_detected' (spesifik uyuşturucu nesneleri) kontrolüne odaklanalım.
        if scores.get('drug', 0) > high_clip_threshold and not drug_related_detected:
            original_drug_score = scores['drug']
            scores['drug'] *= yolo_miss_reduction_factor
            logger.info(f"CLIP 'drug' skoru yüksek ({original_drug_score:.2f}) ama YOLO spesifik onayı yok, skor {scores['drug']:.2f}'ye düşürüldü.")
    
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

# Bu fonksiyonu analysis_service.py tarafından import edilebilmesi için ekliyoruz.
def get_content_analyzer():
    """ContentAnalyzer sınıfından bir örnek (singleton) döndürür."""
    return ContentAnalyzer() 