import insightface 
import numpy as np 
import cv2 
import os 
import torch 
import logging 
from PIL import Image # ERSIN PIL kütüphanesini ekliyoruz
from flask import current_app # ERSIN current_app import edildi
import time 
from typing import Any 

# ERSIN Logger oluştur
logger =logging .getLogger (__name__ )

# ERSIN CustomAgeHead sınıfı (train_v1.py'den alınmalı)
class CustomAgeHead (torch .nn .Module ):
    def __init__ (self ,input_dim =512 ,hidden_dims =[256 ,128 ],output_dim =1 ,input_size =None ):
        super ().__init__ ()
        # ERSIN input_size parametresi varsa onu kullan (backward compatibility için)
        if input_size is not None :
            input_dim =input_size 

        layers =[]
        prev_dim =input_dim 
        for hidden_dim in hidden_dims :
            layers .append (torch .nn .Linear (prev_dim ,hidden_dim ))
            layers .append (torch .nn .ReLU ())
            prev_dim =hidden_dim 
        layers .append (torch .nn .Linear (prev_dim ,output_dim ))
        self .network =torch .nn .Sequential (*layers )

    def forward (self ,x ):
        return self .network (x )

        # ERSIN Versiyonlu model bulucu fonksiyon
def find_latest_age_model (model_path ):
    age_model_dir =os .path .join (model_path ,'models')# ERSIN veya doğrudan model_path altında olabilir
    if not os .path .isdir (age_model_dir ):
        logger .warning (f"Yaş modeli için 'models' klasörü bulunamadı: {age_model_dir }")
        return None 

    model_files =[f for f in os .listdir (age_model_dir )if f .startswith ('age_model_epoch_')and f .endswith ('.pth')]
    if not model_files :
        logger .warning (f"'models' klasöründe özel yaş modeli bulunamadı: {age_model_dir }")
        return None 

        # ERSIN Epoch numarasına göre sırala ve en sonuncuyu al
    model_files .sort (key =lambda x :int (x .split ('_')[-1 ].split ('.')[0 ]),reverse =True )
    latest_model_file =model_files [0 ]
    logger .info (f"Bulunan en son özel yaş modeli: {latest_model_file }")
    return os .path .join (age_model_dir ,latest_model_file )

class InsightFaceAgeEstimator :
    def __init__ (self ,det_size =(1024 ,1024 )):
    # ERSIN Model dosya yolunu ayarla
    # ERSIN active_insightface_path = os.path.join(Config.MODELS_FOLDER, 'age', 'buffalo_l') # Eski yol
        active_insightface_path =current_app .config ['INSIGHTFACE_AGE_MODEL_ACTIVE_PATH']
        base_insightface_path =current_app .config ['INSIGHTFACE_AGE_MODEL_BASE_PATH']

        # ERSIN Insightface modelini yüklemek için kullanılacak asıl yol
        # ERSIN Önce active_model'i kontrol et, eğer boşsa veya gerekli dosyalar yoksa base_model'i kullan.
        # ERSIN insightface
        insightface_root_to_load =active_insightface_path 
        # ERSIN Basit bir kontrol: active_model altında bir şeyler var mı?
        # ERSIN Daha iyi bir kontrol, belirli .onnx dosyalarının varlığını kontrol etmek olabilir.
        # ERSIN detection.onnx yerine buffalo_l modelinin kullandığı det_10g.onnx dosyasını kontrol edelim.
        if not os .path .exists (os .path .join (active_insightface_path ,'det_10g.onnx')):
            logger .warning (f"Aktif InsightFace modeli ({active_insightface_path }) tam değil veya bulunamadı (det_10g.onnx eksik). Base model ({base_insightface_path }) denenecek.")
            insightface_root_to_load =base_insightface_path 
            if not os .path .exists (os .path .join (insightface_root_to_load ,'det_10g.onnx')):
                 logger .error (f"Base InsightFace modeli de ({insightface_root_to_load }) yüklenemiyor. 'det_10g.onnx' bulunamadı.")
                 raise FileNotFoundError (f"InsightFace 'det_10g.onnx' dosyası ne aktif ne de base path'te bulunamadı.")

        logger .info (f"InsightFaceAgeEstimator başlatılıyor. Model için kullanılacak root: {insightface_root_to_load }")

        # ERSIN FaceAnalysis'e geçirilecek directory'nin içeriğini log'la
        if os .path .exists (insightface_root_to_load ):
            logger .info (f"'{insightface_root_to_load }' klasörünün içeriği: {os .listdir (insightface_root_to_load )}")
        else :
            logger .warning (f"'{insightface_root_to_load }' klasörü bulunamadı.")

            # ERSIN FACE_DETECTION_CONFIDENCE değerini config'den oku
            # ERSIN FACTORY_DEFAULTS'taki değer 0.5, kodda kullanılan 0.2 idi.
            # ERSIN Config'den gelen değer öncelikli olacak.
        face_detection_thresh =current_app .config .get ('FACE_DETECTION_CONFIDENCE',0.5 )
        logger .info (f"Kullanılacak yüz tespit eşiği (det_thresh): {face_detection_thresh }")

        # ERSIN Modeli yerel dosyadan yükle
        try :
        # ERSIN Prefer GPU execution provider if available (requires onnxruntime-gpu)
            try :
            # ERSIN ONNX Runtime is an optional dependency, import safely
                import onnxruntime as ort 
                available =set (ort .get_available_providers ()or [])
            except Exception :
                available =set ()

            providers =['CUDAExecutionProvider','CPUExecutionProvider']if 'CUDAExecutionProvider'in available else ['CPUExecutionProvider']
            ctx_id =0 if 'CUDAExecutionProvider'in providers else -1 
            logger .info (f"InsightFace providers: {providers } (ctx_id={ctx_id })")

            self .model =insightface .app .FaceAnalysis (
            name ='buffalo_l',# ERSIN Bu isim, root içindeki alt klasörlerle eşleşebilir veya sadece genel bir addır.
            root =insightface_root_to_load ,# ERSIN Güncellenmiş yol
            providers =providers ,
            det_thresh =face_detection_thresh # ERSIN Dinamik olarak okunan değeri kullan
            )
            self .model .prepare (ctx_id =ctx_id ,det_size =det_size )
            logger .info (f"InsightFace temel modeli başarıyla yüklendi (det_thresh={face_detection_thresh } ile)")
        except Exception as e :
            logger .error (f"InsightFace model yükleme hatası: {str (e )}")
            raise 

            # ERSIN Custom Age Head modelini yükle
        self .device =torch .device ("cuda"if torch .cuda .is_available ()and current_app .config .get ('USE_GPU',True )else "cpu")
        self .custom_age_head =None 

        # ERSIN Performance optimization flags
        self .initialized =True 
        self ._last_cleanup =time .time ()
        self ._memory_threshold_mb =14000 # ERSIN Memory cleanup threshold (14GB - GPU memory'nin %85'i, çok agresif cleanup'ı önler)

        # ERSIN Model load ve initialize tracking için performance
        logger .info (f"InsightFaceAgeEstimator device: {self .device }")

        try :
        # ERSIN Önce active_model'den yüklemeye çalış
            custom_age_head_dir =os .path .join (current_app .config ['MODELS_FOLDER'],'age','custom_age_head','active_model')

            # ERSIN active_model bir sembolik link olabilir, gerçek dizini kontrol et
            if os .path .islink (custom_age_head_dir ):
                custom_age_head_dir =os .path .realpath (custom_age_head_dir )

            if not os .path .exists (custom_age_head_dir ):
            # ERSIN Eğer active_model yoksa base_model'den yükle
                custom_age_head_dir =os .path .join (current_app .config ['MODELS_FOLDER'],'age','custom_age_head','base_model')

            if os .path .exists (custom_age_head_dir ):
            # ERSIN .pth dosyasını bul (model.pth veya custom_age_head.pth olabilir)
                pth_files =[f for f in os .listdir (custom_age_head_dir )if f .endswith ('.pth')]
                if pth_files :
                    model_path =os .path .join (custom_age_head_dir ,pth_files [0 ])
                    logger .info (f"CustomAgeHead model dosyası bulundu: {model_path }")
                    try :
                    # ERSIN Model checkpoint'ini yükle
                        checkpoint =torch .load (model_path ,map_location ='cpu',weights_only =True )

                        # ERSIN Model konfigürasyonunu al
                        if 'model_config'in checkpoint :
                            model_config =checkpoint ['model_config']
                            self .custom_age_head =CustomAgeHead (
                            input_dim =model_config ['input_dim'],
                            hidden_dims =model_config ['hidden_dims'],
                            output_dim =model_config ['output_dim']
                            )
                        else :
                        # ERSIN Varsayılan konfigürasyon
                            self .custom_age_head =CustomAgeHead (input_dim =512 ,hidden_dims =[256 ,128 ],output_dim =1 )

                            # ERSIN Model ağırlıklarını yükle
                        if 'model_state_dict'in checkpoint :
                            self .custom_age_head .load_state_dict (checkpoint ['model_state_dict'])
                        else :
                        # ERSIN Eski formatta kaydedilmiş olabilir
                            self .custom_age_head .load_state_dict (checkpoint )

                        self .custom_age_head .eval ()# ERSIN Evaluation moduna geç
                        self .custom_age_head .to (self .device )
                        logger .info (f"CustomAgeHead başarıyla {model_path } yolundan {self .device } üzerinde yüklendi.")

                        # ERSIN Eski uyumluluk için age_model alias'ı
                        self .age_model =self .custom_age_head 

                    except Exception as e :
                        logger .error (f"CustomAgeHead yüklenirken hata: {str (e )}")
                        self .custom_age_head =None 
                        self .age_model =None 
                else :
                    logger .warning (f"CustomAgeHead model dosyası (.pth) bulunamadı: {custom_age_head_dir }")
                    self .custom_age_head =None 
                    self .age_model =None 
            else :
                logger .warning (f"CustomAgeHead model dizini bulunamadı: {custom_age_head_dir }")
                self .custom_age_head =None 
                self .age_model =None 

        except Exception as e :
            logger .error (f"Custom Age Head model yükleme hatası: {str (e )}")
            self .custom_age_head =None 
            self .age_model =None 

            # ERSIN CLIP modelini yükle - ama önce shared CLIP kontrol et
        self .clip_model =None 
        self .clip_preprocess =None 
        self .tokenizer =None 
        self .clip_device ="cpu"

        logger .info ("⚠️ CLIP yükleme skip edildi - ContentAnalyzer'dan shared CLIP beklenecek")
        logger .info ("🔄 set_shared_clip() metodu ile CLIP inject edilecek")

    def cleanup_models (self ):
        """GPU memory ve model referanslarını temizle - Performance optimization"""
        try :
            logger .info ("InsightFaceAgeEstimator cleanup başlatılıyor...")

            # ERSIN CLIP model temizle
            if hasattr (self ,'clip_model')and self .clip_model is not None :
                del self .clip_model 
                self .clip_model =None 
                logger .debug ("CLIP model cleaned up")

            if hasattr (self ,'clip_preprocess')and self .clip_preprocess is not None :
                del self .clip_preprocess 
                self .clip_preprocess =None 
                logger .debug ("CLIP preprocess cleaned up")

                # ERSIN Custom age head temizle
            if hasattr (self ,'custom_age_head')and self .custom_age_head is not None :
                del self .custom_age_head 
                self .custom_age_head =None 
                logger .debug ("Custom age head cleaned up")

                # ERSIN InsightFace model temizle
            if hasattr (self ,'model')and self .model is not None :
                del self .model 
                self .model =None 
                logger .debug ("InsightFace model cleaned up")

                # ERSIN Tokenizer temizle
            if hasattr (self ,'tokenizer')and self .tokenizer is not None :
                del self .tokenizer 
                self .tokenizer =None 
                logger .debug ("Tokenizer cleaned up")

                # ERSIN GPU cache temizle
            if torch .cuda .is_available ():
                torch .cuda .empty_cache ()
                logger .debug ("GPU cache cleaned up")

                # ERSIN Update cleanup timestamp
            self ._last_cleanup =time .time ()
            logger .info ("InsightFaceAgeEstimator cleanup tamamlandı")

        except Exception as e :
            logger .warning (f"InsightFaceAgeEstimator cleanup sırasında hata: {e }")

    def set_shared_clip (self ,clip_model ,clip_preprocess =None ,tokenizer =None ):
        """
        ContentAnalyzer'dan CLIP modelini paylaş - Memory optimization
        
        Args:
            clip_model: ContentAnalyzer'ın CLIP modeli
            clip_preprocess: CLIP preprocessing fonksiyonu  
            tokenizer: CLIP tokenizer
        """
        try :
            logger .info ("Shared CLIP model InsightFaceAgeEstimator'a inject ediliyor...")

            # ERSIN Mevcut CLIP modelini temizle
            if hasattr (self ,'clip_model')and self .clip_model is not None :
                logger .debug ("Mevcut CLIP model temizleniyor...")
                del self .clip_model 
                self .clip_model =None 

                # ERSIN Shared CLIP modelini ayarla
            self .clip_model =clip_model 
            self .clip_preprocess =clip_preprocess 
            self .tokenizer =tokenizer 

            # ERSIN Device bilgisini güncelle
            if hasattr (clip_model ,'device')and clip_model .device :
                self .clip_device =clip_model .device 
            elif clip_model is not None :
                try :
                # ERSIN Model parametrelerinden device'ı al
                    params =list (clip_model .parameters ())
                    if params :
                        self .clip_device =params [0 ].device 
                    else :
                    # ERSIN Parametre yoksa varsayılan device'ı kullan
                        self .clip_device ="cuda"if torch .cuda .is_available ()else "cpu"
                except Exception as device_err :
                    logger .warning (f"CLIP model device tespit edilemedi: {device_err }, varsayılan kullanılıyor")
                    self .clip_device ="cuda"if torch .cuda .is_available ()else "cpu"
            else :
                self .clip_device ="cpu"

            logger .info (f"✅ Shared CLIP model başarıyla inject edildi! Device: {self .clip_device }")

        except Exception as e :
            logger .error (f"Shared CLIP model inject hatası: {e }")
            self .clip_model =None 
            self .clip_preprocess =None 
            self .tokenizer =None 

    def __del__ (self ):
        """Garbage collection sırasında cleanup yap"""
        try :
            if hasattr (self ,'initialized')and self .initialized :
                self .cleanup_models ()
        except :
            pass # ERSIN Ignore errors during garbage collection

    def _check_memory_usage (self ):
        """Memory usage kontrolü ve otomatik cleanup - Performance monitoring"""
        try :
            current_time =time .time ()
            # ERSIN Her 5 dakikada bir memory kontrolü yap
            if current_time -self ._last_cleanup >300 :# ERSIN 5 minutes

            # ERSIN GPU memory kontrolü
                if torch .cuda .is_available ():
                    gpu_memory_mb =torch .cuda .memory_allocated ()/(1024 *1024 )
                    if gpu_memory_mb >self ._memory_threshold_mb :
                        logger .warning (f"High GPU memory usage detected: {gpu_memory_mb :.1f}MB, triggering cleanup")
                        torch .cuda .empty_cache ()
                        self ._last_cleanup =current_time 

        except Exception as e :
            logger .debug (f"Memory check error: {e }")

    def estimate_age (self ,full_image :np .ndarray [Any ,Any ],face ):
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
        # ERSIN Performance monitoring
        self ._check_memory_usage ()

        if face is None :
            logger .warning ("estimate_age: Geçersiz 'face' nesnesi alındı (None). Varsayılan değerler dönülüyor.")
            return 25.0 ,0.5 ,None 

        logger .info (f"[AGE_LOG] estimate_age başladı. Gelen face bbox: {face .bbox }, Ham InsightFace Yaşı: {face .age }")

        # ERSIN Adım 1: Temel Bilgileri Topla
        embedding_current =face .embedding if hasattr (face ,'embedding')and face .embedding is not None else None 
        age_buffalo_raw =face .age # ERSIN Bu buffalo_l'nin ONNX modelinden gelen ham yaş

        if age_buffalo_raw is None :
            logger .warning ("[AGE_LOG] InsightFace (Buffalo) ham yaşı None, varsayılan (25.0) kullanılacak.")
            age_buffalo_raw =25.0 

        age_buffalo =float (age_buffalo_raw )# ERSIN Tutarlılık için float yapalım

        # ERSIN Adım 1.1: CLIP için Yüz ROI Çıkar
        face_roi =None 
        try :
            x1 ,y1 ,x2 ,y2 =[int (v )for v in face .bbox ]
            h_img ,w_img =full_image .shape [:2 ]
            x1 =max (0 ,x1 )
            y1 =max (0 ,y1 )
            x2 =min (w_img ,x2 )
            y2 =min (h_img ,y2 )
            if x2 >x1 and y2 >y1 :
                 face_roi =full_image [y1 :y2 ,x1 :x2 ]
            else :
                 logger .warning (f"[AGE_LOG] estimate_age: Geçersiz bbox koordinatları nedeniyle face_roi çıkarılamadı: x1={x1 }, y1={y1 }, x2={x2 }, y2={y2 }")
        except Exception as e :
            logger .error (f"[AGE_LOG] face_roi çıkarılırken hata: {str (e )}")

        if face_roi is None :
             logger .warning ("[AGE_LOG] face_roi yok, CLIP tabanlı karşılaştırma yapılamıyor. Buffalo ham tahmini ({age_buffalo:.1f}) ve varsayılan güven (0.5) dönülüyor.")
             # ERSIN Sözde etiket verisi de None olmalı çünkü CLIP güveni yok
             return age_buffalo ,0.5 ,None 

             # ERSIN Adım 2: Buffalo_l Tahmini için CLIP Güvenini Hesapla
        logger .info (f"[AGE_LOG] Buffalo ham tahmini ({age_buffalo :.1f}) için CLIP güveni hesaplanıyor...")
        confidence_clip_buffalo =self ._calculate_confidence_with_clip (face_roi ,age_buffalo )
        logger .info (f"[AGE_LOG] Buffalo Ham Yaşının CLIP Güveni: {confidence_clip_buffalo :.4f}")

        # ERSIN Adım 3: CustomAgeHead Tahmini ve CLIP Güvenini Hesapla (Eğer Mümkünse)
        age_custom =None 
        confidence_clip_custom =-1.0 # ERSIN Karşılaştırmada düşük kalması için başlangıç değeri
        custom_age_calculated =False 

        if self .age_model is not None and embedding_current is not None :
            try :
                with torch .no_grad ():
                    emb_tensor =torch .tensor (embedding_current ,dtype =torch .float32 ).unsqueeze (0 ).to (self .device )
                    # ERSIN NORMALIZE EMBEDDING (Custom model eğitimi sırasında eksik olan adım)
                    emb_tensor =emb_tensor /torch .norm (emb_tensor ,dim =1 ,keepdim =True )
                    age_custom_pred =self .age_model (emb_tensor ).item ()
                logger .info (f"[AGE_LOG] Özel yaş modeli (CustomAgeHead) tahmini: {age_custom_pred :.1f}")
                age_custom =float (age_custom_pred )# ERSIN float yap
                logger .info (f"[AGE_LOG] CustomAgeHead tahmini ({age_custom :.1f}) için CLIP güveni hesaplanıyor...")
                confidence_clip_custom =self ._calculate_confidence_with_clip (face_roi ,age_custom )
                logger .info (f"[AGE_LOG] CustomAgeHead Tahmininin CLIP Güveni: {confidence_clip_custom :.4f}")
                custom_age_calculated =True 
            except Exception as e :
                logger .error (f"[AGE_LOG] Özel yaş modeli (CustomAgeHead) ile tahmin veya CLIP güveni hesaplanırken hata: {str (e )}")
        elif self .age_model is None :
            logger .info ("[AGE_LOG] Özel yaş modeli (CustomAgeHead) yüklenmemiş.")
        elif embedding_current is None :
            logger .info ("[AGE_LOG] Özel yaş modeli (CustomAgeHead) için embedding mevcut değil (face.embedding None).")

            # ERSIN Adım 4: Nihai Yaş ve Güven Belirleme (CLIP GÜVEN SKORLARINA GÖRE)
        final_age =age_buffalo # ERSIN Varsayılan olarak buffalo'nun ham yaşı
        final_confidence =confidence_clip_buffalo # ERSIN ve onun CLIP güveni

        # ERSIN CLIP TABANLI BASİT SEÇİM: CLIP'in hangi tahmine daha yüksek güven verdiğine bak
        if custom_age_calculated :
        # ERSIN CLIP güven skorları çok düşükse (0.15) özel mantık uygula
            LOW_CONFIDENCE_THRESHOLD =0.15 
            both_low_confidence =confidence_clip_buffalo <=LOW_CONFIDENCE_THRESHOLD and confidence_clip_custom <=LOW_CONFIDENCE_THRESHOLD 

            if both_low_confidence and age_custom is not None :
            # ERSIN Her iki model de düşük güvenle tahmin yapıyor
            # ERSIN Çocuklar için daha küçük yaşı tercih et, büyük fark varsa
                age_diff =abs (age_buffalo -age_custom )
                if age_diff >5 :# ERSIN Büyük fark varsa
                # ERSIN Daha küçük yaşı tercih et (çocuklar için daha mantıklı)
                    if age_custom <age_buffalo :
                        final_age =age_custom 
                        final_confidence =confidence_clip_custom 
                        logger .info (f"[AGE_LOG][LOW_CONF_SELECT] Her iki model düşük güven, CustomAgeHead seçildi (daha küçük yaş: {age_custom :.1f} vs {age_buffalo :.1f})")
                    else :
                        final_age =age_buffalo 
                        final_confidence =confidence_clip_buffalo 
                        logger .info (f"[AGE_LOG][LOW_CONF_SELECT] Her iki model düşük güven, Buffalo seçildi (daha küçük yaş: {age_buffalo :.1f} vs {age_custom :.1f})")
                else :
                # ERSIN Fark küçükse, Buffalo'yu tercih et (daha başarılı)
                    final_age =age_buffalo 
                    final_confidence =confidence_clip_buffalo 
                    logger .info (f"[AGE_LOG][LOW_CONF_SELECT] Her iki model düşük güven, Buffalo seçildi (fark küçük: {age_diff :.1f})")
            else :
            # ERSIN Normal seçim: CLIP'in hangi tahmine daha yüksek güven verdiğine bak
                if confidence_clip_custom >confidence_clip_buffalo :
                    final_age =age_custom 
                    final_confidence =confidence_clip_custom 
                    logger .info (f"[AGE_LOG][CLIP_SELECT] CustomAgeHead seçildi (CLIP güveni daha yüksek: {confidence_clip_custom :.4f} > {confidence_clip_buffalo :.4f})")
                elif confidence_clip_custom <confidence_clip_buffalo :
                    final_age =age_buffalo 
                    final_confidence =confidence_clip_buffalo 
                    logger .info (f"[AGE_LOG][CLIP_SELECT] Buffalo seçildi (CLIP güveni daha yüksek: {confidence_clip_buffalo :.4f} > {confidence_clip_custom :.4f})")
                else :
                # ERSIN Eşitlik durumunda: Buffalo'yu tercih et (daha başarılı)
                    final_age =age_buffalo 
                    final_confidence =confidence_clip_buffalo 
                    logger .info (f"[AGE_LOG][CLIP_SELECT] CLIP güven skorları eşit ({confidence_clip_buffalo :.4f}), Buffalo tercih edildi")

                    # ERSIN Adım 5: CustomAgeHead İçin Potansiyel Sözde Etiketli Veri Hazırlama
        pseudo_label_data_to_save =None 
        RECORD_THRESHOLD =current_app .config .get ('PSEUDO_LABEL_RECORD_CLIP_THRESHOLD',0.75 )

        # ERSIN Sözde etiket için buffalo_l'nin kendi ham tahmini ve onun CLIP güvenini kullan
        # ERSIN Dikkat: Burada final_confidence değil, confidence_clip_buffalo kullanılmalı!
        if confidence_clip_buffalo >=RECORD_THRESHOLD :
            logger .info (f"[DATA_LOG] Buffalo ham tahmini (Yaş: {age_buffalo :.1f}, CLIP Güveni: {confidence_clip_buffalo :.4f}) CustomAgeHead için potansiyel eğitim verisi olarak hazırlanıyor (Eşik: {RECORD_THRESHOLD }).")
            bbox_str =",".join (map (str ,[int (v )for v in face .bbox ]))
            emb =embedding_current 
            if emb is not None :
                if hasattr (emb ,'tolist'):
                    emb_str =",".join (str (float (x ))for x in emb .tolist ())
                elif isinstance (emb ,(list ,tuple )):
                    emb_str =",".join (str (float (x ))for x in emb )
                else :
                    emb_str =str (emb )
            else :
                emb_str =None 
            pseudo_label_data_to_save ={
            "face_bbox":bbox_str ,
            "embedding":emb_str ,# ERSIN Artık string olarak
            "pseudo_label_original_age":age_buffalo ,# ERSIN Buffalo'nun ham yaş tahmini
            "pseudo_label_clip_confidence":confidence_clip_buffalo ,# ERSIN Buffalo'nun yaşının CLIP güveni
            "feedback_source":"PSEUDO_BUFFALO_HIGH_CONF",
            "feedback_type":"age_pseudo"
            # ERSIN frame_path, content_id, analysis_id, person_id gibi bilgiler servis katmanında eklenecek
            }
            if embedding_current is None :# ERSIN embedding_current yukarıda zaten None ise buraya girmez ama yine de kontrol
                 logger .warning ("[DATA_LOG] Sözde etiket için embedding (embedding_current) mevcut değil, bu bilgi eksik olacak.")

        logger .info (f"[AGE_LOG][DETAIL] Buffalo yaş tahmini: {age_buffalo :.2f}, CLIP güveni: {confidence_clip_buffalo :.4f}")
        if custom_age_calculated :
            logger .info (f"[AGE_LOG][DETAIL] CustomAgeHead yaş tahmini: {age_custom :.2f}, CLIP güveni: {confidence_clip_custom :.4f}")
        else :
            logger .info (f"[AGE_LOG][DETAIL] CustomAgeHead tahmini yapılamadı.")
        logger .info (f"[AGE_LOG][SELECT] Seçilen yaş tahmini: {final_age :.2f}, CLIP güveni: {final_confidence :.4f}")
        if pseudo_label_data_to_save :
            logger .info (f"[AGE_LOG][PSEUDO] Pseudo label kaydı hazırlanacak: {pseudo_label_data_to_save }")

        logger .info (f"[AGE_LOG] estimate_age tamamlandı. Dönen Nihai Yaş: {final_age :.1f}, Dönen Nihai Güven: {final_confidence :.4f}")
        return final_age ,final_confidence ,pseudo_label_data_to_save 

    def _calculate_confidence_with_clip (self ,face_image ,estimated_age ):
        import time 
        start_time =time .time ()
        logger .info (f"[AGE_LOG] _calculate_confidence_with_clip başladı. Gelen Yaş: {estimated_age :.1f}, Görüntü Shape: {face_image .shape }")
        if self .clip_model is None or face_image .size ==0 :
            logger .warning ("[AGE_LOG] CLIP modeli yok veya yüz görüntüsü geçersiz, varsayılan güven (0.5) dönülüyor.")
            return 0.5 
        try :
        # ERSIN Görüntüyü RGB'ye dönüştür ve PIL formatına çevir
            rgb_image =cv2 .cvtColor (face_image ,cv2 .COLOR_BGR2RGB )
            pil_image =Image .fromarray (rgb_image )

            # ERSIN CLIP için ön işleme
            if self .clip_preprocess is None :
                return 0.5 
            preprocessed_image =self .clip_preprocess (pil_image ).unsqueeze (0 ).to (self .clip_device )

            # ERSIN DİREKT YAŞ SORUSU: "this face X years old"
            # ERSIN Not: CLIP sayısal yaşta sınırlı; 18 eşiğinde (under18 vs adult) daha hassas olması için
            # ERSIN karşıt prompt setini 17–21 bandına yakınlaştırıyoruz ve ayrıca under18 vs adult promptları ile
            # ERSIN ek bir ayrım skoru logluyoruz.
            age =int (round (estimated_age ))

            # ERSIN Spesifik yaş sorusu
            target_prompt =f"this face is {age } years old"

            def _uniq_ints (values ):
                out =[]
                seen =set ()
                for v in values :
                    try :
                        iv =int (v )
                    except Exception :
                        continue 
                    if iv in seen :
                        continue 
                    if iv <1 or iv >90 :
                        continue 
                    if iv ==age :
                        continue 
                    out .append (iv )
                    seen .add (iv )
                return out 

                # ERSIN Karşıt yaş soruları (farklı yaş aralıklarından)
                # ERSIN 18 eşiğine yakın yaşlarda ayrımı güçlendirmek için 17–21 bandını dahil et.
            if 13 <=age <=22 :
                opposing_ages =_uniq_ints (
                [
                age -4 ,
                age -2 ,
                17 ,
                18 ,
                19 ,
                20 ,
                21 ,
                age +2 ,
                age +4 ,
                8 ,
                30 ,
                50 ,
                ]
                )[:8 ]
            elif age <10 :
                opposing_ages =_uniq_ints ([25 ,45 ,65 ,16 ])# ERSIN Bebek/çocuk için yetişkin yaşları
            elif age <20 :
                opposing_ages =_uniq_ints ([5 ,30 ,50 ,70 ])# ERSIN Genç için diğer yaşlar
            elif age <30 :
                opposing_ages =_uniq_ints ([8 ,45 ,65 ,15 ])# ERSIN Genç yetişkin için diğer yaşlar
            elif age <50 :
                opposing_ages =_uniq_ints ([10 ,20 ,65 ,75 ])# ERSIN Orta yaş için diğer yaşlar
            else :
                opposing_ages =_uniq_ints ([8 ,18 ,30 ,45 ])# ERSIN Yaşlı için genç yaşlar

            opposing_prompts =[f"this face is {opp_age } years old"for opp_age in opposing_ages ]

            # ERSIN Under18 vs Adult (18+) ayrımı için ek prompt seti (confidence doğrulama için)
            under18_prompts =[
            "this face is under 18 years old",
            "this person is a minor (under 18)",
            "a teenage person (under 18)",
            ]
            adult_prompts =[
            "this face is 18 years old or older",
            "this person is an adult (18 or older)",
            "an adult person (18+)",
            ]

            # ERSIN Tüm prompt'ları birleştir (tek encode_text çağrısı için)
            all_prompts =[target_prompt ]+opposing_prompts +under18_prompts +adult_prompts 

            # ERSIN CLIP ile benzerlik hesapla
            with torch .no_grad ():
                image_features =self .clip_model .encode_image (preprocessed_image )
                image_features /=image_features .norm (dim =-1 ,keepdim =True )

                if self .tokenizer is None :
                    return 0.5 
                text_inputs =self .tokenizer (all_prompts ).to (self .clip_device )
                text_features =self .clip_model .encode_text (text_inputs )
                text_features /=text_features .norm (dim =-1 ,keepdim =True )

                # ERSIN Benzerlik skorlarını al
                similarities =(100.0 *image_features @text_features .T ).squeeze (0 ).cpu ().numpy ()

                # ERSIN similarities array'inin boş olmadığını kontrol et
            if similarities is None or len (similarities )==0 :
                logger .warning ("CLIP similarities array boş, varsayılan güven skoru dönülüyor")
                return 0.5 

            target_score =float (similarities [0 ])
            opposing_scores =similarities [1 :1 +len (opposing_prompts )]
            avg_opposing =float (np .mean (opposing_scores ))
            max_opposing =float (np .max (opposing_scores ))

            # ERSIN MAXIMUM opposing score ile karşılaştır (daha hassas)
            score_diff =target_score -max_opposing 

            # ERSIN Eğer target score, max opposing'den düşükse net negatif güven
            if score_diff <0 :
                confidence_score =0.1 # ERSIN Minimum güven
            else :
            # ERSIN Softmax-style confidence
                confidence_score =1.0 /(1.0 +np .exp (-score_diff *2 ))
                confidence_score =max (0.1 ,min (0.9 ,confidence_score ))

                # ERSIN Under18 vs Adult için ek skor (log + 18 eşiğinde stabilizasyon)
            u_start =1 +len (opposing_prompts )
            u_end =u_start +len (under18_prompts )
            a_start =u_end 
            a_end =a_start +len (adult_prompts )

            under18_scores =similarities [u_start :u_end ]
            adult_scores =similarities [a_start :a_end ]
            under18_mean =float (np .mean (under18_scores ))
            adult_mean =float (np .mean (adult_scores ))
            under18_diff =under18_mean -adult_mean 
            prob_under18 =float (1.0 /(1.0 +np .exp (-under18_diff *0.8 )))
            side_conf =prob_under18 if age <18 else (1.0 -prob_under18 )

            # ERSIN 18 bandında (13–22) confidence'i under18 ekseni ile harmanla
            if 13 <=age <=22 :
                confidence_score =(confidence_score *0.5 )+(side_conf *0.5 )
                confidence_score =max (0.1 ,min (0.9 ,float (confidence_score )))

            end_time =time .time ()
            elapsed_time =end_time -start_time 

            logger .info (f"[AGE_LOG] DİREKT YAŞ SORUSU - Target Yaş: {age }")
            logger .info (f"[AGE_LOG] Target Prompt: '{target_prompt }'")
            logger .info (f"[AGE_LOG] Opposing Prompts: {opposing_prompts }")
            logger .info (f"[AGE_LOG] Target Skor: {target_score :.4f}")
            logger .info (f"[AGE_LOG] Opposing Skorlar: {[f'{s :.4f}'for s in opposing_scores ]}")
            logger .info (f"[AGE_LOG] Opposing Ort: {avg_opposing :.4f}, Max: {max_opposing :.4f}")
            logger .info (f"[AGE_LOG] Skor Farkı (Target - Max): {score_diff :.4f}")
            logger .info (
            f"[AGE_LOG] Under18 vs Adult: under18_mean={under18_mean :.4f} adult_mean={adult_mean :.4f} "
            f"diff={under18_diff :.4f} prob_under18={prob_under18 :.3f} side_conf={side_conf :.3f}"
            )
            logger .info (f"[AGE_LOG] Final Güven: {confidence_score :.4f}")
            logger .info (f"[AGE_LOG] CLIP hesaplama süresi: {elapsed_time :.3f} saniye")

            return confidence_score 

        except Exception as e :
            end_time =time .time ()
            elapsed_time =end_time -start_time 
            logger .error (f"[AGE_LOG] CLIP ile güven skoru hesaplanırken hata: {str (e )} (Süre: {elapsed_time :.3f}s)")
            return 0.5 # ERSIN Hata durumunda varsayılan güven

    def compute_face_encoding (self ,face_image :np .ndarray [Any ,Any ]):
        """
        Verilen yüz görüntüsünden embedding (vektör) çıkarır.
        Args:
            face_image: BGR (OpenCV) formatında numpy array
        Returns:
            embedding: np.ndarray veya None
        """
        if self .model is None :
            return None 
        faces =self .model .get (face_image )
        if not faces :
            return None 
        return faces [0 ].embedding 

    def compare_faces (self ,encoding1 ,encoding2 ,tolerance =0.6 ):
        """
        İki embedding (yüz vektörü) arasındaki benzerliği kontrol eder.
        Args:
            encoding1: np.ndarray
            encoding2: np.ndarray
            tolerance: float (daha düşük değer = daha sıkı eşleşme)
        Returns:
            bool: Benzerse True
        """
        if encoding1 is None or encoding2 is None :
            return False 
        distance =np .linalg .norm (np .array (encoding1 )-np .array (encoding2 ))
        return distance <=tolerance 

    def get_faces (self ,image :np .ndarray [Any ,Any ]):
    # ERSIN This method is not provided in the original file or the code block
    # ERSIN It's assumed to exist as it's called in the estimate_age method
        pass 

        # ERSIN Kullanım örneği:
        # ERSIN estimator = InsightFaceAgeEstimator()
        # ERSIN img = cv2.imread('face.jpg')
        # ERSIN age = estimator.estimate_age(img)
        # ERSIN print('Tahmini yaş:', age)

        # ERSIN Bu fonksiyonu analysis_service.py tarafından import edilebilmesi için ekliyoruz.
def get_age_estimator ():
    """InsightFaceAgeEstimator sınıfından bir örnek döndürür."""
    return InsightFaceAgeEstimator ()