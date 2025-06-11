import insightface
import numpy as np
import cv2
import os
import torch
import re
import logging
from config import Config
from PIL import Image  # PIL kütüphanesini ekliyoruz
import math
from flask import current_app # current_app import edildi
import open_clip # ADDED IMPORT
import time

# Logger oluştur
logger = logging.getLogger(__name__)

# CustomAgeHead sınıfı (train_v1.py'den alınmalı)
class CustomAgeHead(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[256, 128], output_dim=1, input_size=None):
        super().__init__()
        # input_size parametresi varsa onu kullan (backward compatibility için)
        if input_size is not None:
            input_dim = input_size
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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
        # active_insightface_path = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_l') # Eski yol
        active_insightface_path = current_app.config['INSIGHTFACE_AGE_MODEL_ACTIVE_PATH']
        base_insightface_path = current_app.config['INSIGHTFACE_AGE_MODEL_BASE_PATH']

        # Insightface modelini yüklemek için kullanılacak asıl yol
        # Önce active_model'i kontrol et, eğer boşsa veya gerekli dosyalar yoksa base_model'i kullan.
        # insightface.app.FaceAnalysis, root parametresinde model dosyalarını (örn: detection.onnx, genderage.onnx) bekler.
        insightface_root_to_load = active_insightface_path
        # Basit bir kontrol: active_model altında bir şeyler var mı?
        # Daha iyi bir kontrol, belirli .onnx dosyalarının varlığını kontrol etmek olabilir.
        # detection.onnx yerine buffalo_l modelinin kullandığı det_10g.onnx dosyasını kontrol edelim.
        if not os.path.exists(os.path.join(active_insightface_path, 'det_10g.onnx')):
            logger.warning(f"Aktif InsightFace modeli ({active_insightface_path}) tam değil veya bulunamadı (det_10g.onnx eksik). Base model ({base_insightface_path}) denenecek.")
            insightface_root_to_load = base_insightface_path
            if not os.path.exists(os.path.join(insightface_root_to_load, 'det_10g.onnx')):
                 logger.error(f"Base InsightFace modeli de ({insightface_root_to_load}) yüklenemiyor. 'det_10g.onnx' bulunamadı.")
                 raise FileNotFoundError(f"InsightFace 'det_10g.onnx' dosyası ne aktif ne de base path'te bulunamadı.")

        logger.info(f"InsightFaceAgeEstimator başlatılıyor. Model için kullanılacak root: {insightface_root_to_load}")
        
        # Log the contents of the directory that will be passed to FaceAnalysis
        if os.path.exists(insightface_root_to_load):
            logger.info(f"'{insightface_root_to_load}' klasörünün içeriği: {os.listdir(insightface_root_to_load)}")
        else:
            logger.warning(f"'{insightface_root_to_load}' klasörü bulunamadı.")

        # FACE_DETECTION_CONFIDENCE değerini config'den oku
        # FACTORY_DEFAULTS'taki değer 0.5, kodda kullanılan 0.2 idi.
        # Config'den gelen değer öncelikli olacak.
        face_detection_thresh = current_app.config.get('FACE_DETECTION_CONFIDENCE', 0.5) 
        logger.info(f"Kullanılacak yüz tespit eşiği (det_thresh): {face_detection_thresh}")

        # Modeli yerel dosyadan yükle
        try:
            self.model = insightface.app.FaceAnalysis(
                name='buffalo_l', # Bu isim, root içindeki alt klasörlerle eşleşebilir veya sadece genel bir addır.
                root=insightface_root_to_load, # Güncellenmiş yol
                providers=['CPUExecutionProvider'],
                det_thresh=face_detection_thresh # Dinamik olarak okunan değeri kullan
            )
            self.model.prepare(ctx_id=0, det_size=det_size)
            logger.info(f"InsightFace temel modeli başarıyla yüklendi (det_thresh={face_detection_thresh} ile)")
        except Exception as e:
            logger.error(f"InsightFace model yükleme hatası: {str(e)}")
            raise
        
        # Custom Age Head modelini yükle
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        self.custom_age_head = None
        
        # Performance optimization flags
        self.initialized = True
        self._last_cleanup = time.time()
        self._memory_threshold_mb = 500  # Memory cleanup threshold
        
        # Model load and initialize tracking for performance
        logger.info(f"InsightFaceAgeEstimator device: {self.device}")
        
        try:
            # Önce active_model'den yüklemeye çalış
            custom_age_head_dir = os.path.join(current_app.config['MODELS_FOLDER'], 'age', 'custom_age_head', 'active_model')
            
            # active_model bir sembolik link olabilir, gerçek dizini kontrol et
            if os.path.islink(custom_age_head_dir):
                custom_age_head_dir = os.path.realpath(custom_age_head_dir)
            
            if not os.path.exists(custom_age_head_dir):
                # Eğer active_model yoksa base_model'den yükle
                custom_age_head_dir = os.path.join(current_app.config['MODELS_FOLDER'], 'age', 'custom_age_head', 'base_model')
            
            if os.path.exists(custom_age_head_dir):
                # .pth dosyasını bul (model.pth veya custom_age_head.pth olabilir)
                pth_files = [f for f in os.listdir(custom_age_head_dir) if f.endswith('.pth')]
                if pth_files:
                    model_path = os.path.join(custom_age_head_dir, pth_files[0])
                    logger.info(f"CustomAgeHead model dosyası bulundu: {model_path}")
                    try:
                        # Model checkpoint'ini yükle
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                        
                        # Model konfigürasyonunu al
                        if 'model_config' in checkpoint:
                            model_config = checkpoint['model_config']
                            self.custom_age_head = CustomAgeHead(
                                input_dim=model_config['input_dim'],
                                hidden_dims=model_config['hidden_dims'],
                                output_dim=model_config['output_dim']
                            )
                        else:
                            # Varsayılan konfigürasyon
                            self.custom_age_head = CustomAgeHead(input_dim=512, hidden_dims=[256, 128], output_dim=1)
                        
                        # Model ağırlıklarını yükle
                        if 'model_state_dict' in checkpoint:
                            self.custom_age_head.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            # Eski formatta kaydedilmiş olabilir
                            self.custom_age_head.load_state_dict(checkpoint)
                        
                        self.custom_age_head.eval()  # Evaluation moduna geç
                        self.custom_age_head.to(self.device)
                        logger.info(f"CustomAgeHead başarıyla {model_path} yolundan {self.device} üzerinde yüklendi.")
                        
                        # Eski uyumluluk için age_model alias'ı
                        self.age_model = self.custom_age_head
                        
                    except Exception as e:
                        logger.error(f"CustomAgeHead yüklenirken hata: {str(e)}")
                        self.custom_age_head = None
                        self.age_model = None
                else:
                    logger.warning(f"CustomAgeHead model dosyası (.pth) bulunamadı: {custom_age_head_dir}")
                    self.custom_age_head = None
                    self.age_model = None
            else:
                logger.warning(f"CustomAgeHead model dizini bulunamadı: {custom_age_head_dir}")
                self.custom_age_head = None
                self.age_model = None
                
        except Exception as e:
            logger.error(f"Custom Age Head model yükleme hatası: {str(e)}")
            self.custom_age_head = None
            self.age_model = None
            
        # CLIP modelini yükle
        try:
            device = "cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu"
            self.clip_device = device

            # ORIJINAL YÖNTEM: Önce base model yükle (dfn5b), sonra fine-tuned weights
            logger.info(f"CLIP modeli yükleniyor: ViT-H-14-378-quickgelu, Device: {device}")
            
            try:
                model, _, preprocess_val = open_clip.create_model_and_transforms(
                    'ViT-H-14-378-quickgelu', 
                    pretrained="dfn5b",
                    device=self.clip_device
                )
                
                # Fine-tuned model varsa yükle
                try:
                    active_model_path = current_app.config['OPENCLIP_MODEL_ACTIVE_PATH']
                    model_file_path = os.path.join(active_model_path, 'open_clip_pytorch_model.bin')
                    
                    if os.path.exists(model_file_path):
                        logger.info(f"Fine-tuned CLIP weights yükleniyor (yaş tahmini için): {model_file_path}")
                        checkpoint = torch.load(model_file_path, map_location=self.clip_device)
                        model.load_state_dict(checkpoint, strict=False)
                        logger.info("Fine-tuned CLIP weights başarıyla yüklendi! (yaş tahmini)")
                    else:
                        logger.info("Fine-tuned CLIP weights bulunamadı, base model kullanılıyor (yaş tahmini)")
                        
                except Exception as ft_error:
                    logger.warning(f"Fine-tuned weights yükleme hatası (yaş tahmini): {str(ft_error)}")
                    logger.info("Base model ile devam ediliyor... (yaş tahmini)")
                
                model.eval()
                self.clip_model = model
                self.clip_preprocess = preprocess_val
                logger.info(f"✅ CLIP modeli (yaş tahmini için) başarıyla yüklendi! Device: {self.clip_device}")
                
            except Exception as clip_error:
                logger.warning(f"CLIP model (yaş için) yüklenemedi: {clip_error}")
                # Fallback: CLIP olmadan devam et
                self.clip_model = None
                self.clip_preprocess = None
                logger.warning("CLIP modeli olmadan güven skoru 0.5 olarak sabitlenecek")

            # Tokenizer'ı yükle (OpenCLIP için) - sadece CLIP model başarılıysa
            if self.clip_model is not None:
                logger.info("OpenCLIP tokenizer (ViT-H-14-378-quickgelu) yaş tahmini için yükleniyor...")
                self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
                logger.info("OpenCLIP tokenizer (yaş tahmini) başarıyla yüklendi.")

        except Exception as e:
            logger.error(f"CLIP modeli yüklenemedi: {str(e)}")
            logger.warning("CLIP modeli olmadan güven skoru 0.5 olarak sabitlenecek")
            self.clip_model = None
            self.clip_preprocess = None

    def cleanup_models(self):
        """GPU memory ve model referanslarını temizle - Performance optimization"""
        try:
            logger.info("InsightFaceAgeEstimator cleanup başlatılıyor...")
            
            # CLIP model temizle
            if hasattr(self, 'clip_model') and self.clip_model is not None:
                del self.clip_model
                self.clip_model = None
                logger.debug("CLIP model cleaned up")
            
            if hasattr(self, 'clip_preprocess') and self.clip_preprocess is not None:
                del self.clip_preprocess
                self.clip_preprocess = None
                logger.debug("CLIP preprocess cleaned up")
                
            # Custom age head temizle
            if hasattr(self, 'custom_age_head') and self.custom_age_head is not None:
                del self.custom_age_head
                self.custom_age_head = None
                logger.debug("Custom age head cleaned up")
                
            # InsightFace model temizle
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                logger.debug("InsightFace model cleaned up")
                
            # Tokenizer temizle
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                logger.debug("Tokenizer cleaned up")
                
            # GPU cache temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleaned up")
                
            # Update cleanup timestamp
            self._last_cleanup = time.time()
            logger.info("InsightFaceAgeEstimator cleanup tamamlandı")
            
        except Exception as e:
            logger.warning(f"InsightFaceAgeEstimator cleanup sırasında hata: {e}")
    
    def __del__(self):
        """Garbage collection sırasında cleanup yap"""
        try:
            if hasattr(self, 'initialized') and self.initialized:
                self.cleanup_models()
        except:
            pass  # Ignore errors during garbage collection

    def _check_memory_usage(self):
        """Memory usage kontrolü ve otomatik cleanup - Performance monitoring"""
        try:
            current_time = time.time()
            # Her 5 dakikada bir memory kontrolü yap
            if current_time - self._last_cleanup > 300:  # 5 minutes
                
                # GPU memory kontrolü
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    if gpu_memory_mb > self._memory_threshold_mb:
                        logger.warning(f"High GPU memory usage detected: {gpu_memory_mb:.1f}MB, triggering cleanup")
                        torch.cuda.empty_cache()
                        self._last_cleanup = current_time
                
        except Exception as e:
            logger.debug(f"Memory check error: {e}")

    def estimate_age(self, full_image: np.ndarray, face):
        """
        Verilen 'face' nesnesi için yaş tahminini ve CLIP güven skorunu döndürür.
        Yüz tespiti bu fonksiyonda *yapılmaz*, önceden tespit edilmiş face nesnesi kullanılır.

        Args:
            full_image (np.ndarray): Yüzün bulunduğu orijinal tam kare (BGR).
            face: InsightFace modelinin get() metodundan dönen yüz nesnesi.

        Returns:
            Tuple: (final_age, final_confidence, pseudo_label_data_to_save)
                   pseudo_label_data_to_save bir dict veya None olabilir.
        """
        # Performance monitoring
        self._check_memory_usage()
        
        if face is None:
            logger.warning("estimate_age: Geçersiz 'face' nesnesi alındı (None). Varsayılan değerler dönülüyor.")
            return 25.0, 0.5, None

        logger.info(f"[AGE_LOG] estimate_age başladı. Gelen face bbox: {face.bbox}, Ham InsightFace Yaşı: {face.age}")

        # Adım 1: Temel Bilgileri Topla
        embedding_current = face.embedding if hasattr(face, 'embedding') and face.embedding is not None else None
        age_buffalo_raw = face.age # Bu buffalo_l'nin ONNX modelinden gelen ham yaş

        if age_buffalo_raw is None:
            logger.warning("[AGE_LOG] InsightFace (Buffalo) ham yaşı None, varsayılan (25.0) kullanılacak.")
            age_buffalo_raw = 25.0
        
        age_buffalo = float(age_buffalo_raw) # Tutarlılık için float yapalım

        # Adım 1.1: CLIP için Yüz ROI Çıkar
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

        if face_roi is None:
             logger.warning("[AGE_LOG] face_roi yok, CLIP tabanlı karşılaştırma yapılamıyor. Buffalo ham tahmini ({age_buffalo:.1f}) ve varsayılan güven (0.5) dönülüyor.")
             # Sözde etiket verisi de None olmalı çünkü CLIP güveni yok
             return age_buffalo, 0.5, None

        # Adım 2: Buffalo_l Tahmini için CLIP Güvenini Hesapla
        logger.info(f"[AGE_LOG] Buffalo ham tahmini ({age_buffalo:.1f}) için CLIP güveni hesaplanıyor...")
        confidence_clip_buffalo = self._calculate_confidence_with_clip(face_roi, age_buffalo)
        logger.info(f"[AGE_LOG] Buffalo Ham Yaşının CLIP Güveni: {confidence_clip_buffalo:.4f}")

        # Adım 3: CustomAgeHead Tahmini ve CLIP Güvenini Hesapla (Eğer Mümkünse)
        age_custom = None
        confidence_clip_custom = -1.0 # Karşılaştırmada düşük kalması için başlangıç değeri
        custom_age_calculated = False

        if self.age_model is not None and embedding_current is not None:
            try:
                with torch.no_grad():
                    emb_tensor = torch.tensor(embedding_current, dtype=torch.float32).unsqueeze(0)
                    # NORMALIZE EMBEDDING (Custom model eğitimi sırasında eksik olan adım)
                    emb_tensor = emb_tensor / torch.norm(emb_tensor, dim=1, keepdim=True)
                    age_custom_pred = self.age_model(emb_tensor).item()
                logger.info(f"[AGE_LOG] Özel yaş modeli (CustomAgeHead) tahmini: {age_custom_pred:.1f}")
                age_custom = float(age_custom_pred) # float yap
                logger.info(f"[AGE_LOG] CustomAgeHead tahmini ({age_custom:.1f}) için CLIP güveni hesaplanıyor...")
                confidence_clip_custom = self._calculate_confidence_with_clip(face_roi, age_custom)
                logger.info(f"[AGE_LOG] CustomAgeHead Tahmininin CLIP Güveni: {confidence_clip_custom:.4f}")
                custom_age_calculated = True
            except Exception as e:
                logger.error(f"[AGE_LOG] Özel yaş modeli (CustomAgeHead) ile tahmin veya CLIP güveni hesaplanırken hata: {str(e)}")
        elif self.age_model is None:
            logger.info("[AGE_LOG] Özel yaş modeli (CustomAgeHead) yüklenmemiş.")
        elif embedding_current is None:
            logger.info("[AGE_LOG] Özel yaş modeli (CustomAgeHead) için embedding mevcut değil (face.embedding None).")

        # Adım 4: Nihai Yaş ve Güven Belirleme (ÇAPRAZ TEST İLE)
        final_age = age_buffalo # Varsayılan olarak buffalo'nun ham yaşı
        final_confidence = confidence_clip_buffalo # ve onun CLIP güveni
        
        # ÇAPRAZ TEST SİSTEMİ: Her iki tahmin için de karşıt soruları sor
        if custom_age_calculated:
            logger.info(f"[AGE_LOG][CROSS_TEST] Çapraz test başlıyor...")
            
            # Normal çapraz test
            # Buffalo'nun tahmini için Custom'ın yaş sorusunu sor
            buffalo_cross_confidence = self._calculate_confidence_with_clip(face_roi, age_custom)
            
            # Custom'ın tahmini için Buffalo'nun yaş sorusunu sor  
            custom_cross_confidence = self._calculate_confidence_with_clip(face_roi, age_buffalo)
            
            logger.info(f"[AGE_LOG][CROSS_TEST] Buffalo {age_buffalo:.1f} yaş tahmini, Custom'ın {age_custom:.1f} yaş sorusunda: {buffalo_cross_confidence:.4f}")
            logger.info(f"[AGE_LOG][CROSS_TEST] Custom {age_custom:.1f} yaş tahmini, Buffalo'nun {age_buffalo:.1f} yaş sorusunda: {custom_cross_confidence:.4f}")
            
            # Net üstünlük kontrolü
            buffalo_net_score = confidence_clip_buffalo - buffalo_cross_confidence
            custom_net_score = confidence_clip_custom - custom_cross_confidence
            
            logger.info(f"[AGE_LOG][CROSS_TEST] Buffalo Net Skor: {buffalo_net_score:.4f} (kendi: {confidence_clip_buffalo:.4f} - karşıt: {buffalo_cross_confidence:.4f})")
            logger.info(f"[AGE_LOG][CROSS_TEST] Custom Net Skor: {custom_net_score:.4f} (kendi: {confidence_clip_custom:.4f} - karşıt: {custom_cross_confidence:.4f})")
            
            # En yüksek net skora sahip olan kazansın
            if custom_net_score > buffalo_net_score:
                final_age = age_custom
                final_confidence = confidence_clip_custom
                logger.info(f"[AGE_LOG][CROSS_SELECT] Çapraz test sonucu: CustomAgeHead seçildi (Net: {custom_net_score:.4f})")
            else:
                final_age = age_buffalo  
                final_confidence = confidence_clip_buffalo
                logger.info(f"[AGE_LOG][CROSS_SELECT] Çapraz test sonucu: Buffalo seçildi (Net: {buffalo_net_score:.4f})")
        
        # Adım 5: CustomAgeHead İçin Potansiyel Sözde Etiketli Veri Hazırlama
        pseudo_label_data_to_save = None
        RECORD_THRESHOLD = current_app.config.get('PSEUDO_LABEL_RECORD_CLIP_THRESHOLD', 0.75) 

        # Sözde etiket için buffalo_l'nin kendi ham tahmini ve onun CLIP güvenini kullan
        # Dikkat: Burada final_confidence değil, confidence_clip_buffalo kullanılmalı!
        if confidence_clip_buffalo >= RECORD_THRESHOLD:
            logger.info(f"[DATA_LOG] Buffalo ham tahmini (Yaş: {age_buffalo:.1f}, CLIP Güveni: {confidence_clip_buffalo:.4f}) CustomAgeHead için potansiyel eğitim verisi olarak hazırlanıyor (Eşik: {RECORD_THRESHOLD}).")
            bbox_str = ",".join(map(str, [int(v) for v in face.bbox])) 
            emb = embedding_current
            if emb is not None:
                if hasattr(emb, 'tolist'):
                    emb_str = ",".join(str(float(x)) for x in emb.tolist())
                elif isinstance(emb, (list, tuple)):
                    emb_str = ",".join(str(float(x)) for x in emb)
                else:
                    emb_str = str(emb)
            else:
                emb_str = None
            pseudo_label_data_to_save = {
                "face_bbox": bbox_str,
                "embedding": emb_str, # Artık string olarak
                "pseudo_label_original_age": age_buffalo, # Buffalo'nun ham yaş tahmini
                "pseudo_label_clip_confidence": confidence_clip_buffalo, # Buffalo'nun yaşının CLIP güveni
                "feedback_source": "PSEUDO_BUFFALO_HIGH_CONF",
                "feedback_type": "age_pseudo"
                # frame_path, content_id, analysis_id, person_id gibi bilgiler servis katmanında eklenecek
            }
            if embedding_current is None: # embedding_current yukarıda zaten None ise buraya girmez ama yine de kontrol
                 logger.warning("[DATA_LOG] Sözde etiket için embedding (embedding_current) mevcut değil, bu bilgi eksik olacak.")

        logger.info(f"[AGE_LOG][DETAIL] Buffalo yaş tahmini: {age_buffalo:.2f}, CLIP güveni: {confidence_clip_buffalo:.4f}")
        if custom_age_calculated:
            logger.info(f"[AGE_LOG][DETAIL] CustomAgeHead yaş tahmini: {age_custom:.2f}, CLIP güveni: {confidence_clip_custom:.4f}")
        else:
            logger.info(f"[AGE_LOG][DETAIL] CustomAgeHead tahmini yapılamadı.")
        if custom_age_calculated and confidence_clip_custom >= confidence_clip_buffalo:
            logger.info(f"[AGE_LOG][SELECT] Seçilen yaş tahmini: CustomAgeHead (Yaş: {age_custom:.2f}, Güven: {confidence_clip_custom:.4f})")
        else:
            logger.info(f"[AGE_LOG][SELECT] Seçilen yaş tahmini: Buffalo (Yaş: {age_buffalo:.2f}, Güven: {confidence_clip_buffalo:.4f})")
        if pseudo_label_data_to_save:
            logger.info(f"[AGE_LOG][PSEUDO] Pseudo label kaydı hazırlanacak: {pseudo_label_data_to_save}")

        logger.info(f"[AGE_LOG] estimate_age tamamlandı. Dönen Nihai Yaş: {final_age:.1f}, Dönen Nihai Güven: {final_confidence:.4f}")
        return final_age, final_confidence, pseudo_label_data_to_save

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
            
            # DİREKT YAŞ SORUSU: "this face is X years old"
            age = int(round(estimated_age))
            
            # Spesifik yaş sorusu
            target_prompt = f"this face is {age} years old"
            
            # Karşıt yaş soruları (farklı yaş aralıklarından)
            opposing_ages = []
            if age < 10:
                opposing_ages = [25, 45, 65, 16]  # Bebek/çocuk için yetişkin yaşları
            elif age < 20:
                opposing_ages = [5, 30, 50, 70]   # Genç için diğer yaşlar
            elif age < 30:
                opposing_ages = [8, 45, 65, 15]   # Genç yetişkin için diğer yaşlar
            elif age < 50:
                opposing_ages = [10, 20, 65, 75]  # Orta yaş için diğer yaşlar
            else:
                opposing_ages = [8, 18, 30, 45]   # Yaşlı için genç yaşlar
            
            opposing_prompts = [f"this face is {opp_age} years old" for opp_age in opposing_ages]
            
            # Tüm prompt'ları birleştir
            all_prompts = [target_prompt] + opposing_prompts
            
            # CLIP ile benzerlik hesapla
            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                text_inputs = self.tokenizer(all_prompts).to(self.clip_device)
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Benzerlik skorlarını al
                similarities = (100.0 * image_features @ text_features.T).squeeze(0).cpu().numpy()
            
            target_score = float(similarities[0])
            opposing_scores = similarities[1:]
            avg_opposing = float(np.mean(opposing_scores))
            max_opposing = float(np.max(opposing_scores))
            
            # MAXIMUM opposing score ile karşılaştır (daha hassas)
            score_diff = target_score - max_opposing
            
            # Eğer target score, max opposing'den düşükse net negatif güven
            if score_diff < 0:
                confidence_score = 0.1  # Minimum güven
            else:
                # Softmax-style confidence
                confidence_score = 1.0 / (1.0 + np.exp(-score_diff * 2))
                confidence_score = max(0.1, min(0.9, confidence_score))
            
            logger.info(f"[AGE_LOG] DİREKT YAŞ SORUSU - Target Yaş: {age}")
            logger.info(f"[AGE_LOG] Target Prompt: '{target_prompt}'")
            logger.info(f"[AGE_LOG] Opposing Prompts: {opposing_prompts}")
            logger.info(f"[AGE_LOG] Target Skor: {target_score:.4f}")
            logger.info(f"[AGE_LOG] Opposing Skorlar: {[f'{s:.4f}' for s in opposing_scores]}")
            logger.info(f"[AGE_LOG] Opposing Ort: {avg_opposing:.4f}, Max: {max_opposing:.4f}")
            logger.info(f"[AGE_LOG] Skor Farkı (Target - Max): {score_diff:.4f}")
            logger.info(f"[AGE_LOG] Final Güven: {confidence_score:.4f}")
            
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