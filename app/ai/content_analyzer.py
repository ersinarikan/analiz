# pyright: reportImportCycles=false
# ERSIN Flask uygulamalarında circular import'lar yaygın ve genelde sorun yaratmaz
import os 
import numpy as np 
import cv2 
import logging 
import json 
from flask import current_app 
import warnings 
warnings .filterwarnings ('ignore',category =DeprecationWarning )
from ultralytics import YOLO 
import torch 
import open_clip 
from PIL import Image # ERSIN CLIP için PIL gerekiyor
import time 
import threading 
from typing import Any ,cast
from app .utils .serialization_utils import convert_numpy_types_to_python 

logger =logging .getLogger (__name__ )

# ERSIN thread-safe cache kilidi
_cache_lock =threading .Lock ()
# ERSIN thread-safe model cache
_models_cache ={}

# ERSIN ensemble servisini import et
from app .services .ensemble_integration_service import get_ensemble_service 

# ERSIN Kullanılmayan global _models_cache kaldırıldı

class ContentAnalyzer :
    """
    İçerik analiz sınıfı, görüntülerdeki şiddet, yetişkin içeriği, vb. kategorileri tespit eder.
    - YOLO, OpenCLIP ve diğer modelleri kullanır.
    - Thread-safe singleton pattern ile çalışır.
    """

    # ERSIN thread-safe singleton implementation
    _instance =None 
    _lock =threading .Lock ()

    def __new__ (cls )->'ContentAnalyzer':
        """Thread-safe singleton pattern implementasyonu"""
        if cls ._instance is None :
            with cls ._lock :
            # ERSIN çift kontrol kilitleme pattern
                if cls ._instance is None :
                    logger .info ("Yeni ContentAnalyzer singleton instance oluşturuluyor...")
                    start_time =time .time ()
                    cls ._instance =super (ContentAnalyzer ,cls ).__new__ (cls )
                    cls ._instance .initialized =False 
                    load_time =time .time ()-start_time 
                    logger .info (f"ContentAnalyzer singleton instance oluşturuldu ({load_time :.2f}s)")
        return cls ._instance 

    @classmethod 
    def reset_instance (cls ):
        """Singleton instance'ını thread-safe şekilde sıfırlar ve model cache'ini temizler"""
        global _models_cache 

        with cls ._lock :
            with _cache_lock :
            # ERSIN GPU memory temizle
                if cls ._instance and hasattr (cls ._instance ,'cleanup_models'):
                    cls ._instance .cleanup_models ()

                    # ERSIN Cache temizle
                _models_cache .clear ()

                # ERSIN Instance'ı sıfırla
                cls ._instance =None 
                logger .info ("ContentAnalyzer instance ve model cache thread-safe şekilde sıfırlandı")

    def cleanup_models (self ):
        """GPU memory ve model referanslarını temizle"""
        try :
        # ERSIN CLIP model temizle
            if hasattr (self ,'clip_model'):
                del self .clip_model 
            if hasattr (self ,'clip_preprocess'):
                del self .clip_preprocess 

                # ERSIN YOLO model temizle
            if hasattr (self ,'yolo_model'):
                del self .yolo_model 

                # ERSIN Tokenizer temizle
            if hasattr (self ,'tokenizer'):
                del self .tokenizer 

                # ERSIN GPU cache temizle
            if torch .cuda .is_available ():
                torch .cuda .empty_cache ()

            logger .info ("Model cleanup tamamlandı")

        except Exception as e :
            logger .warning (f"Model cleanup sırasında hata: {e }")

    def __del__ (self ):
        """Garbage collection sırasında GPU memory temizle"""
        self .cleanup_models ()

    def __init__ (self ):
        """
        ContentAnalyzer sınıfını başlatır ve gerekli modelleri yükler.
        Bu sınıf, şiddet, yetişkin içeriği, taciz, silah, madde kullanımı tespiti için kullanılır.
        """
        # ERSIN Eğer zaten başlatıldıysa tekrar başlatma
        if self .initialized :
            return 

        try :
        # ERSIN Sanity mode: offline prompt değerlendirme hafif olmalı ve GPU OOM önlenmeli
        # ERSIN Etkinleştirildiğinde YOLO ve ensemble atlanır, sadece CLIP scoring yapılır
            self .sanity_mode =(os .environ .get ("WSANALIZ_SANITY_MODE","")or "").strip ().lower ()in {"1","true","yes","y","on"}

            # ERSIN Model klasörünü belirle
            model_folder =current_app .config .get ('MODELS_FOLDER',os .path .join (os .path .dirname (os .path .abspath (__file__ )),'..','..','storage','models'))

            # ERSIN YOLOv8 modelini yükle - nesne tespiti için hala faydalı
            if self .sanity_mode :
                self .yolo_model =None 
                logger .info ("WSANALIZ_SANITY_MODE=1 -> YOLO yükleme atlandı (CLIP-only)")
            else :
                self .yolo_model =self ._load_yolo_model (model_folder )

                # ERSIN CLIP modelini yükle, daha büyük ve doğru versiyonu seç
                # ERSIN Tensor'lar için tek device kaynağı tut, CPU/GPU uyumsuzluğunu önle
            self .clip_model ,self .clip_preprocess ,self .device =self ._load_clip_model ()

            # ERSIN Tokenizer'ı yükle (OpenCLIP için)
            logger .info ("OpenCLIP tokenizer (ViT-H-14-378-quickgelu) yükleniyor...")
            self .tokenizer =open_clip .get_tokenizer ('ViT-H-14-378-quickgelu')
            logger .info ("OpenCLIP tokenizer başarıyla yüklendi.")

            # ERSIN Eğitilmiş classification head'i yükle (varsa)
            self .classification_head =self ._load_classification_head ()

            # ERSIN Ensemble service'i yükle
            if self .sanity_mode :
                self .ensemble_service =None 
            else :
                try :
                    self .ensemble_service =get_ensemble_service ()
                    logger .info ("Ensemble service başarıyla yüklendi")
                except Exception as e :
                    logger .warning (f"Ensemble service yüklenemedi: {str (e )}")
                    self .ensemble_service =None 

                    # ERSIN Prompt template'leri (template ensembling)
                    # ERSIN OPTIMIZED: CLIP için daha etkili template'ler (performans için 5'ten 3'e düşürüldü)
                    # ERSIN Not: ağırlıklandırma yok; template'ler eşit ağırlıklı ortalanır.
                    # ERSIN En etkili 3 template seçildi (performans/doğruluk dengesi)
            self .prompt_templates =[
            "a photo of {x}",
            "an image of {x}",
            "{x}",
            ]

            # ERSIN Kategori tanımlayıcıları - Pozitif ve zıt (negatif) promptlar
            # ERSIN OPTIMIZED: Daha spesifik, görsel olarak ayırt edilebilir prompt'lar
            self .category_prompts ={
            "violence":{
            "positive":[
            # ERSIN Görsel olarak somut şiddet sahneleri
            "two people physically fighting with fists",
            "a person punching another person in the face",
            "a person being hit and falling down",
            "blood on someone's face from violence",
            "a violent physical altercation between people",
            "someone being attacked and injured",
            "a street fight with multiple people brawling",
            "visible injuries from physical violence",
            "aggressive physical confrontation"
            ],
            "negative":[
            # ERSIN Şiddet değil: spor, tartışma, normal etkileşim
            "two people having a verbal argument",
            "professional boxing match in a ring",
            "martial arts training or competition",
            "people playing sports together",
            "friendly physical interaction",
            "helping someone stand up",
            "a handshake or greeting",
            "people exercising or working out together"
            ]
            },
            "adult_content":{
            "positive":[
            # ERSIN Görsel olarak net yetişkin içerik - EN ETKİLİ prompt'lar seçildi (performans için)
            "explicit sexual content with genitals visible",
            "pornographic scene with sexual activity",
            "naked person in sexual pose",
            "explicit sexual intercourse",
            "genitalia exposed in sexual context",
            "sexual act being performed",
            "explicit nudity in sexual situation",
            # ERSIN NSFW düşükken CLIP'in daha iyi tespit etmesi için en etkili 3 prompt
            "nude person engaging in sexual activity",
            "explicit sexual content showing private parts",
            "adult pornography with visible genitals"
            ],
            "negative":[
            # ERSIN Yetişkin içerik ile karışan benign sahneler - güçlü negative'ler
            "person wearing normal clothes in everyday setting",
            "family photo with people fully clothed",
            "beach scene with people in swimwear",
            "athlete in sports uniform or workout clothes",
            "person in underwear for fashion or fitness",
            "medical examination with professional context",
            "artistic nude photography in gallery",
            "baby or child in normal clothing",
            "people in casual everyday clothing",
            "non-sexual context with minimal clothing"
            ]
            },
            "harassment":{
            "positive":[
            # ERSIN Görsel olarak tespit edilebilir taciz davranışları
            "person touching another person inappropriately",
            "unwanted groping or sexual touching",
            "person's hand on someone's body without consent",
            "inappropriate touching in crowded place",
            "sexual harassment by physical contact",
            "person being touched against their will",
            "non-consensual physical contact"
            ],
            "negative":[
            # ERSIN Taciz değil: normal temas, şiddet, yardım
            "two people shaking hands politely",
            "friendly hug between people",
            "person helping someone stand up",
            "a physical fight with punching",
            "people exercising together",
            "normal social interaction",
            "respectful physical contact"
            ]
            },
            "weapon":{
            "positive":[
            # ERSIN Görsel olarak net silah tespiti - EN ETKİLİ prompt'lar seçildi (performans için)
            "person holding a handgun or pistol clearly visible",
            "person pointing a gun at someone with finger on trigger",
            "rifle or firearm clearly visible in person's hand",
            "knife being used as a weapon in threatening manner",
            "person armed with a visible firearm or handgun",
            "weapon being brandished or shown in hand",
            "firearm in threatening context with person holding it",
            # ERSIN En spesifik ve etkili 3 prompt
            "handgun or pistol clearly visible in person's hand",
            "person holding a real firearm or gun",
            "visible gun barrel or pistol grip in hand"
            ],
            "negative":[
            # ERSIN Silah değil: mutfak, alet, oyuncak, telefon, cüzdan (GÜÇLENDİRİLMİŞ)
            "kitchen knife on cutting board",
            "person cooking with kitchen tools",
            "tool being used for work or construction",
            "toy gun or fake weapon",
            "sports equipment like baseball bat",
            "peaceful scene with no weapons",
            "harmless tool in workshop",
            # ERSIN Telefon, cüzdan gibi nesneler için ek negatif prompt'lar
            "person holding a cell phone or mobile device",
            "person with wallet or phone in hand",
            "rectangular object like phone or wallet in hand",
            "person holding electronic device or phone",
            "small rectangular object in person's pocket or hand",
            "person using mobile phone or smartphone",
            "harmless everyday object like phone or wallet"
            ]
            },
            "drug":{
            "positive":[
            # ERSIN Görsel olarak tespit edilebilir uyuşturucu kullanımı
            "person injecting drugs with syringe",
            "drug paraphernalia like pipes or bongs",
            "person smoking marijuana or joint",
            "illegal drug use visible",
            "drugs being prepared or consumed",
            "syringe used for drug injection",
            "drug-related equipment and substances"
            ],
            "negative":[
            # ERSIN Uyuşturucu değil: tıbbi, normal içecek, ilaç
            "medical syringe in hospital or clinic",
            "person receiving vaccination",
            "people drinking beverages normally",
            "medicine bottles in pharmacy",
            "healthy lifestyle and exercise",
            "normal food and drink consumption",
            "medical equipment in clinical setting"
            ]
            }
            }
            # ERSIN Kategori text tokenları önceden hesapla (tek prompt) - DEBUG FIX
            self .category_text_features ={}
            try :
                for category ,prompts in self .category_prompts .items ():
                    logger .info (f"Text features hazırlanıyor: {category }")
                    # ERSIN Template ensembling var; burada debug için ilk template + ilk positive kullanıyoruz
                    seed_prompt =self .prompt_templates [0 ].format (x =prompts ["positive"][0 ])
                    device =getattr (self ,"device",("cuda"if torch .cuda .is_available ()else "cpu"))
                    text_input =self .tokenizer (seed_prompt ).to (device )
                    with torch .no_grad ():
                    # ERSIN OpenCLIP encode_text() exists but type checker doesn't recognize it
                        encode_text_method =getattr (self .clip_model ,'encode_text',None )
                        if encode_text_method is not None and callable (encode_text_method ):
                            text_feature =encode_text_method (text_input )
                        else :
                            raise RuntimeError ("CLIP model encode_text method not available")
                        # ERSIN Type checker için text_feature'ı torch.Tensor olarak cast et
                        if text_feature is not None :
                            text_feature_tensor =cast (torch .Tensor ,text_feature )
                            text_feature =text_feature_tensor /text_feature_tensor .norm (dim =-1 ,keepdim =True )
                            if text_feature is not None :
                                self .category_text_features [category ]=text_feature [0 ]# ERSIN Tek vektör
                logger .info (f"Text features hazırlandı: {list (self .category_text_features .keys ())}")
            except Exception as text_feature_error :
                logger .warning (f"Text features hazırlanamadı: {text_feature_error }")
                self .category_text_features ={}

                # ERSIN NSFW model lazy loading (sadece gerektiğinde yüklenecek)
            self ._nsfw_model =None 
            self ._nsfw_model_loaded =False 
            self ._nsfw_preprocess =None 

            self .initialized =True 
            logger .info ("ContentAnalyzer - CLIP modeli başarıyla yüklendi")
        except Exception as e :
            logger .error (f"Content analyzer model yükleme hatası: {str (e )}")
            self .initialized =False 
            raise 

    def _load_clip_model (self ):
        """Thread-safe CLIP model yükleme"""
        cache_key ="clip_model"

        # ERSIN thread-safe cache kontrolü
        with _cache_lock :
            if cache_key in _models_cache :
                logger .info ("CLIP modeli thread-safe cache'den kullanılıyor")
                return _models_cache [cache_key ]

        try :
            device ="cuda"if torch .cuda .is_available ()and current_app .config .get ('USE_GPU',True )else "cpu"
            logger .info (f"CLIP modeli yükleniyor: ViT-H-14-378-quickgelu, Device: {device }")

            # ERSIN ORIJINAL YÖNTEM: Önce base model yükle (dfn5b), sonra fine-tuned weights
            model ,_ ,preprocess_val =open_clip .create_model_and_transforms (
            'ViT-H-14-378-quickgelu',
            pretrained ="dfn5b",
            device =device 
            )

            # ERSIN Fine-tuned model varsa yükle
            try :
                active_model_path =current_app .config ['OPENCLIP_MODEL_ACTIVE_PATH']
                model_file_path =os .path .join (active_model_path ,'open_clip_pytorch_model.bin')

                if os .path .exists (model_file_path ):
                    logger .info (f"Fine-tuned CLIP weights yükleniyor: {model_file_path }")
                    checkpoint =torch .load (model_file_path ,map_location =device )
                    # ERSIN PyTorch load_state_dict() exists but type checker doesn't recognize it
                    load_state_dict_method =getattr (model ,'load_state_dict',None )
                    if load_state_dict_method is not None and callable (load_state_dict_method ):
                        load_state_dict_method (checkpoint ,strict =False )
                    else :
                        raise RuntimeError ("Model load_state_dict method not available")
                    logger .info ("Fine-tuned CLIP weights başarıyla yüklendi!")
                else :
                    logger .info ("Fine-tuned CLIP weights bulunamadı, base model kullanılıyor")

            except Exception as ft_error :
                logger .warning (f"Fine-tuned weights yükleme hatası: {str (ft_error )}")
                logger .info ("Base model ile devam ediliyor...")

                # ERSIN PyTorch eval() exists but type checker doesn't recognize it
            eval_method =getattr (model ,'eval',None )
            if eval_method is not None and callable (eval_method ):
                eval_method ()
            else :
                raise RuntimeError ("Model eval method not available")

                # ERSIN thread-safe cache'e kaydet
            with _cache_lock :
                _models_cache [cache_key ]=(model ,preprocess_val ,device )

            logger .info (f"✅ CLIP modeli başarıyla yüklendi! Device: {device }")
            return model ,preprocess_val ,device 

        except Exception as e :
            logger .error (f"CLIP model yükleme hatası: {str (e )}")
            raise e 

    def _load_yolo_model (self ,model_folder ):
        """Thread-safe YOLOv8 model yükleme"""
        cache_key ="yolov8"

        # ERSIN thread-safe cache kontrolü
        with _cache_lock :
            if cache_key in _models_cache :
                logger .info ("YOLOv8 modeli thread-safe cache'den kullanılıyor")
                return _models_cache [cache_key ]

                # ERSIN Model path belirleme
        active_yolo_model_base_path =current_app .config ['YOLO_MODEL_ACTIVE_PATH']
        yolo_model_filename =current_app .config .get ('YOLO_MODEL_NAME','yolov8x')+'.pt'
        yolo_model_full_path =os .path .join (active_yolo_model_base_path ,yolo_model_filename )

        if not os .path .exists (yolo_model_full_path ):
            logger .warning (f"Aktif YOLO modeli bulunamadı: {yolo_model_full_path }. Base model denenecek.")
            base_yolo_path =current_app .config ['YOLO_MODEL_BASE_PATH']
            yolo_model_full_path =os .path .join (base_yolo_path ,yolo_model_filename )

            if not os .path .exists (yolo_model_full_path ):
                logger .error (f"YOLO modeli ne aktif ne de base path'te bulunamadı: {yolo_model_full_path }")
                # ERSIN Fallback: Online'dan indirme
                try :
                    logger .info (f"YOLO modeli online'dan indirilmeye çalışılıyor: {yolo_model_filename }")
                    model =YOLO (yolo_model_filename )

                    # ERSIN thread-safe cache'e kaydet
                    with _cache_lock :
                        _models_cache [cache_key ]=model 

                    logger .info (f"{yolo_model_filename } modeli online'dan yüklendi.")
                    return model 
                except Exception as fallback_err :
                    logger .error (f"Fallback YOLO modeli online'dan yüklenemedi: {fallback_err }")
                    raise FileNotFoundError (f"YOLO modeli bulunamadı ve online'dan indirilemedi: {yolo_model_full_path }")

        try :
            logger .info (f"YOLOv8 modeli yükleniyor: {yolo_model_full_path }")
            model =YOLO (yolo_model_full_path )

            # ERSIN thread-safe cache'e kaydet
            with _cache_lock :
                _models_cache [cache_key ]=model 

            logger .info (f"YOLOv8 modeli başarıyla yüklendi: {yolo_model_full_path }")
            return model 

        except Exception as yolo_err :
            logger .error (f"YOLOv8 modeli yüklenemedi: {str (yolo_err )}")
            raise 

    def _load_nsfw_model (self ):
        """Thread-safe NSFW model lazy loading (ONNX Runtime)"""
        if self ._nsfw_model_loaded and self ._nsfw_model is not None :
            return self ._nsfw_model 

        try :
        # ERSIN Config erişimi: Önce current_app, yoksa config modülünden, yoksa environment variable
            nsfw_enabled =False 
            model_path =None 

            try :
                from flask import current_app 
                nsfw_enabled =current_app .config .get ('NSFW_ENABLED',False )
                model_path =current_app .config .get ('NSFW_MODEL_PATH')
            except RuntimeError :
            # ERSIN Application context yok, config modülünden veya environment variable'dan al
                try :
                    from config import Config 
                    nsfw_enabled =Config .NSFW_ENABLED 
                    model_path =Config .NSFW_MODEL_PATH 
                except (ImportError ,AttributeError ):
                # ERSIN Config modülü de yüklenemezse environment variable'dan al
                # ERSIN os zaten dosyanın başında import edilmiş
                    nsfw_enabled =os .environ .get ('NSFW_ENABLED','False').lower ()in ('true','1','t')
                    model_path =os .environ .get ('NSFW_MODEL_PATH')or os .path .join (
                    os .environ .get ('MODELS_FOLDER','storage/models'),'nsfw','nsfw-detector-224.onnx'
                    )

                    # ERSIN NSFW devre dışı mı kontrol et
            if not nsfw_enabled :
                logger .info ("NSFW modeli devre dışı (NSFW_ENABLED=False)")
                return None 
            if not model_path or not os .path .exists (model_path ):
                logger .warning (f"NSFW model dosyası bulunamadı: {model_path }")
                return None 

            logger .info (f"NSFW modeli yükleniyor: {model_path }")

            # ERSIN ONNX Runtime ile model yükle
            try :
                import onnxruntime as ort 
            except ImportError :
                logger .error ("onnxruntime yüklü değil. Lütfen 'pip install onnxruntime-gpu' veya 'pip install onnxruntime' yükleyin.")
                return None 

                # ERSIN Provider seçimi (CUDA varsa öncelikli)
            providers =['CUDAExecutionProvider','CPUExecutionProvider']if torch .cuda .is_available ()else ['CPUExecutionProvider']

            session_options =ort .SessionOptions ()
            session_options .graph_optimization_level =ort .GraphOptimizationLevel .ORT_ENABLE_ALL 

            ort_session =ort .InferenceSession (
            str (model_path ),
            sess_options =session_options ,
            providers =providers 
            )

            # ERSIN Metadata yükle
            metadata_path =os .path .join (os .path .dirname (model_path ),'metadata.json')
            metadata ={}
            if os .path .exists (metadata_path ):
                with open (metadata_path ,'r')as f :
                    metadata =json .load (f )

                    # ERSIN Preprocess fonksiyonu hazırla
            input_size =metadata .get ('input_size',384 )
            normalization =metadata .get ('normalization',{
            'mean':[0.485 ,0.456 ,0.406 ],
            'std':[0.229 ,0.224 ,0.225 ]
            })

            def nsfw_preprocess (image ):
                """PIL Image'ı NSFW model input formatına dönüştür"""
                from PIL import Image 
                import numpy as np 

                # ERSIN Resize
                image =image .resize ((input_size ,input_size ),Image .Resampling .LANCZOS )

                # ERSIN PIL Image'ı numpy array'e çevir
                img_array =np .array (image ).astype (np .float32 )/255.0 

                # ERSIN Normalize
                mean =np .array (normalization ['mean']).reshape (1 ,1 ,3 )
                std =np .array (normalization ['std']).reshape (1 ,1 ,3 )
                img_array =(img_array -mean )/std 

                # ERSIN HWC -> CHW
                img_array =np .transpose (img_array ,(2 ,0 ,1 ))

                # ERSIN Batch dimension ekle
                img_array =np .expand_dims (img_array ,axis =0 )

                return img_array .astype (np .float32 )

            self ._nsfw_model =ort_session 
            self ._nsfw_preprocess =nsfw_preprocess 
            self ._nsfw_model_loaded =True 

            logger .info (f"✅ NSFW modeli başarıyla yüklendi (Provider: {ort_session .get_providers ()})")
            return self ._nsfw_model 

        except Exception as e :
            logger .error (f"NSFW model yükleme hatası: {e }",exc_info =True )
            self ._nsfw_model =None 
            self ._nsfw_model_loaded =False 
            return None 

    def _load_classification_head (self ):
        """Eğitilmiş classification head'i yükle (varsa)"""
        try :
        # ERSIN Aktif model versiyonun classification head'ini kontrol et
            active_model_path =current_app .config ['OPENCLIP_MODEL_ACTIVE_PATH']
            classifier_path =os .path .join (active_model_path ,'classification_head.pth')

            if os .path .exists (classifier_path ):
                import torch .nn as nn 
                device ="cuda"if torch .cuda .is_available ()and current_app .config .get ('USE_GPU',True )else "cpu"

                # ERSIN Classification head yapısını oluştur
                # ERSIN OpenCLIP visual attribute exists but type checker doesn't recognize it
                visual =getattr (self .clip_model ,'visual',None )
                if visual is None :
                    raise RuntimeError ("CLIP model visual not available")
                feature_dim =getattr (visual ,'output_dim',512 )
                classifier =nn .Sequential (
                nn .Linear (feature_dim ,512 ),
                nn .ReLU (),
                nn .Dropout (0.3 ),
                nn .Linear (512 ,256 ),
                nn .ReLU (),
                nn .Dropout (0.3 ),
                nn .Linear (256 ,5 ),# ERSIN 5 kategori: violence, adult_content, harassment, weapon, drug
                nn .Sigmoid ()
                ).to (device )

                # ERSIN Ağırlıkları yükle
                classifier .load_state_dict (torch .load (classifier_path ,map_location =device ))
                # ERSIN PyTorch eval() exists but type checker doesn't recognize it
                eval_method =getattr (classifier ,'eval',None )
                if eval_method is not None and callable (eval_method ):
                    eval_method ()
                else :
                    raise RuntimeError ("Classifier eval method not available")

                logger .info (f"Eğitilmiş classification head yüklendi: {classifier_path }")
                return classifier 
            else :
                logger .info ("Eğitilmiş classification head bulunamadı, prompt-based yaklaşım kullanılacak")
                return None 

        except Exception as e :
            logger .warning (f"Classification head yüklenirken hata: {str (e )}, prompt-based yaklaşım kullanılacak")
            return None 

    def _get_clip_adult_score_only (self ,image_features ,pil_image )->float :
        """
        Sadece adult_content için CLIP skorunu hesapla (hibrit yaklaşım için)
        
        Args:
            image_features: CLIP image features (zaten hesaplanmış)
            pil_image: PIL Image (prompt encoding için gerekli değil, ama tutarlılık için)
            
        Returns:
            float: CLIP adult_content skoru (0.0-1.0)
        """
        try :
            cat ="adult_content"

            # ERSIN Template ensembling
            pos_prompts =[
            tpl .format (x =p )
            for p in self .category_prompts [cat ]["positive"]
            for tpl in self .prompt_templates 
            ]
            neg_prompts =[
            tpl .format (x =p )
            for p in self .category_prompts [cat ]["negative"]
            for tpl in self .prompt_templates 
            ]
            all_prompts =pos_prompts +neg_prompts 

            device =getattr (self ,"device",("cuda"if torch .cuda .is_available ()else "cpu"))
            text_inputs =self .tokenizer (all_prompts ).to (device )
            with torch .no_grad ():
                encode_text_method =getattr (self .clip_model ,'encode_text',None )
                if encode_text_method is not None and callable (encode_text_method ):
                    text_features =encode_text_method (text_inputs )
                else :
                    raise RuntimeError ("CLIP model encode_text method not available")
                # ERSIN Type checker için text_features'ı torch.Tensor olarak cast et
                if text_features is not None :
                    text_features_tensor =cast (torch .Tensor ,text_features )
                    text_features =text_features_tensor /text_features_tensor .norm (dim =-1 ,keepdim =True )
                if text_features is not None and image_features is not None :
                    text_features_t =cast (torch .Tensor ,text_features )
                    image_features_t =cast (torch .Tensor ,image_features )
                    similarities =(image_features_t @text_features_t .T ).squeeze (0 ).cpu ().numpy ()
                else :
                    similarities =np .array ([])

            if len (similarities )==0 :
                return 0.0 

            pos_sims =similarities [:len (pos_prompts )]
            pos_score =float (np .percentile (pos_sims ,90 ))if len (pos_sims )>0 else 0.0 

            neg_sims =similarities [len (pos_prompts ):]
            neg_score =float (np .percentile (neg_sims ,90 ))if len (neg_sims )>0 else 0.0 

            fark =pos_score -neg_score 

            # ERSIN Score calculation (adult_content için özel squashing)
            import math 
            SQUASH_FACTOR =18.0 # ERSIN adult_content için
            squashed_fark =math .tanh (fark *SQUASH_FACTOR )
            base_score =max (0.0 ,min (1.0 ,float (max (0.0 ,squashed_fark ))))

            return base_score 

        except Exception as e :
            logger .error (f"CLIP adult_content skoru hesaplama hatası: {e }",exc_info =True )
            return 0.0 

    def _analyze_with_nsfw_model (self ,image_path :str |np .ndarray [Any ,Any ])->float :
        """
        NSFW modeli ile görüntü analizi yapar
        
        Args:
            image_path: Görüntü yolu veya numpy array
            
        Returns:
            float: NSFW skoru (0.0-1.0)
        """
        try :
        # ERSIN NSFW modeli yükle (lazy loading)
            nsfw_model =self ._load_nsfw_model ()
            if nsfw_model is None :
                logger .warning ("NSFW modeli yüklenemedi, 0.0 döndürülüyor")
                return 0.0 

                # ERSIN Görüntüyü yükle
            if isinstance (image_path ,str ):
                pil_image =Image .open (image_path ).convert ("RGB")
            else :
            # ERSIN numpy array'den PIL Image'a çevir
                cv_rgb =cv2 .cvtColor (image_path ,cv2 .COLOR_BGR2RGB )
                pil_image =Image .fromarray (cv_rgb )

                # ERSIN Preprocess
            if self ._nsfw_preprocess is None :
                logger .error ("NSFW preprocess fonksiyonu yüklenemedi")
                return 0.0 

            preprocessed =self ._nsfw_preprocess (pil_image )

            # ERSIN Inference
            outputs =nsfw_model .run (None ,{'input':preprocessed })

            # ERSIN Output'u [0, 1] aralığına normalize et
            # ERSIN NSFW modeli binary classification, output logits formatında: [SFW_logit, NSFW_logit]
            # ERSIN ONNX Runtime output'unu numpy array'e çevir
            output_array =np .array (outputs [0 ])
            if len (output_array .shape )>1 and output_array .shape [1 ]>1 :
            # ERSIN Binary classification: [SFW_logit, NSFW_logit]
            # ERSIN Index 0 = SFW, Index 1 = NSFW
                sfw_logit =float (output_array [0 ][0 ])
                nsfw_logit =float (output_array [0 ][1 ])

                # ERSIN Softmax uygula (binary classification için doğru yöntem)
                import math 
                exp_sfw =math .exp (sfw_logit )
                exp_nsfw =math .exp (nsfw_logit )
                total =exp_sfw +exp_nsfw 
                nsfw_score =exp_nsfw /total # ERSIN NSFW probability
            else :
            # ERSIN Single output: NSFW probability veya logit
                nsfw_score =float (output_array [0 ][0 ])if len (output_array [0 ])>0 else 0.0 
                # ERSIN Eğer logit ise sigmoid uygula
                if nsfw_score <0 or nsfw_score >1 :
                    import math 
                    nsfw_score =1.0 /(1.0 +math .exp (-nsfw_score ))

                    # ERSIN [0, 1] aralığına clamp
            nsfw_score =max (0.0 ,min (1.0 ,nsfw_score ))

            logger .debug (f"[NSFW_INFERENCE] NSFW skoru: {nsfw_score :.4f}")
            return nsfw_score 

        except Exception as e :
            logger .error (f"NSFW inference hatası: {e }",exc_info =True )
            return 0.0 

    def analyze_image (self ,image_path :str |np .ndarray [Any ,Any ],content_id :str |None =None ,person_id :str |None =None )->tuple [float ,float ,float ,float ,float ,float ,list [dict [str ,Any ]]]:
        """
        Bir resmi CLIP modeli ile analiz eder ve içerik skorlarını hesaplar.
        (YENİ: Ensemble desteği ile)
        """
        try :
        # ERSIN OpenCV ile görüntüyü yükle (YOLO için)
            if isinstance (image_path ,str ):
                cv_image =cv2 .imread (image_path )
                if cv_image is None :
                    raise ValueError (f"Resim yüklenemedi: {image_path }")
                if os .path .exists (image_path ):
                    pil_image =Image .open (image_path ).convert ("RGB")
                else :
                    cv_rgb =cv2 .cvtColor (cv_image ,cv2 .COLOR_BGR2RGB )
                    pil_image =Image .fromarray (cv_rgb )
            else :
                cv_image =image_path 
                cv_rgb =cv2 .cvtColor (cv_image ,cv2 .COLOR_BGR2RGB )
                pil_image =Image .fromarray (cv_rgb )

                # ERSIN YOLOv8 ile nesne tespiti (sanity-mode'da kapalı: GPU/CPU overhead ve OOM azalt)
            detected_objects =[]
            yolo_model =getattr (self ,"yolo_model",None )
            if yolo_model is not None and not getattr (self ,"sanity_mode",False ):
                results =yolo_model (cv_image )
                for r in results :
                    boxes =r .boxes 
                    if boxes is None or len (boxes )==0 :
                        continue 
                    for box in boxes :
                    # ERSIN Box tensor'larının boş olup olmadığını kontrol et
                        if box .xyxy is None or len (box .xyxy )==0 :
                            continue 
                        if box .conf is None or len (box .conf )==0 :
                            continue 
                        if box .cls is None or len (box .cls )==0 :
                            continue 

                        try :
                            x1 ,y1 ,x2 ,y2 =box .xyxy [0 ].cpu ().numpy ()
                            x1 ,y1 ,x2 ,y2 =int (x1 ),int (y1 ),int (x2 ),int (y2 )
                            w ,h =x2 -x1 ,y2 -y1 
                            conf =float (box .conf [0 ].cpu ().numpy ())
                            cls_id =int (box .cls [0 ].cpu ().numpy ())
                            label =yolo_model .names [cls_id ]
                            detected_objects .append ({'label':label ,'confidence':conf ,'box':[x1 ,y1 ,w ,h ]})
                        except (IndexError ,AttributeError ,ValueError )as e :
                            logger .warning (f"YOLO box işleme hatası: {e }, box atlanıyor")
                            continue 

                            # ERSIN CLIP ile görüntü özelliklerini çıkar
            device =getattr (self ,"device",("cuda"if torch .cuda .is_available ()else "cpu"))
            clip_preprocess =getattr (self ,"clip_preprocess",None )
            if clip_preprocess is None :
                raise RuntimeError ("CLIP preprocess not initialized")
            preprocessed_image =clip_preprocess (pil_image ).unsqueeze (0 ).to (device )
            with torch .no_grad ():
            # ERSIN OpenCLIP encode_image() exists but type checker doesn't recognize it
                encode_image_method =getattr (self .clip_model ,'encode_image',None )
                if encode_image_method is not None and callable (encode_image_method ):
                    image_features =encode_image_method (preprocessed_image )
                else :
                    raise RuntimeError ("CLIP model encode_image method not available")
                # ERSIN Type checker için image_features'ı torch.Tensor olarak cast et
                image_features_tensor =cast (torch .Tensor ,image_features )
                image_features =image_features_tensor /image_features_tensor .norm (dim =-1 ,keepdim =True )

                # ERSIN NSFW kontrolü ve adult_content kategorisini çıkar (image_features hazır olduktan sonra)
                # ERSIN Config erişimi: Önce current_app, yoksa config modülünden, yoksa environment variable
            nsfw_enabled =False 
            nsfw_threshold =0.3 

            try :
                from flask import current_app 
                nsfw_enabled =current_app .config .get ('NSFW_ENABLED',False )
                nsfw_threshold =current_app .config .get ('NSFW_THRESHOLD',0.3 )
            except RuntimeError :
            # ERSIN Application context yok, config modülünden veya environment variable'dan al
                try :
                    from config import Config 
                    nsfw_enabled =Config .NSFW_ENABLED 
                    nsfw_threshold =Config .NSFW_THRESHOLD 
                except (ImportError ,AttributeError ):
                # ERSIN Config modülü de yüklenemezse environment variable'dan al
                    nsfw_enabled =os .environ .get ('NSFW_ENABLED','False').lower ()in ('true','1','t')
                    nsfw_threshold =float (os .environ .get ('NSFW_THRESHOLD','0.3'))

            nsfw_score =0.0 
            clip_adult_score =0.0 # ERSIN CLIP skorunu sadece NSFW düşükse hesapla

            categories =list (self .category_prompts .keys ())
            logger .info (f"[CATEGORIES_INIT] Başlangıç kategorileri: {categories }")

            # ERSIN NSFW etkinleştird ise adult_content'i categories listesinden çıkar
            if nsfw_enabled :
            # ERSIN Önce NSFW inference yap
                nsfw_score =self ._analyze_with_nsfw_model (image_path )
                # ERSIN nsfw_threshold zaten yukarıda alındı

                # ERSIN NSFW düşük skor verdiysa CLIP'e sor (hibrit yaklaşım)
                if nsfw_score <nsfw_threshold :
                # ERSIN NSFW bulamadı, CLIP'e sor (sadece adult_content için)
                # ERSIN Not: CLIP image encoding zaten yapıldı, sadece adult_content prompt'larını çalıştır
                    clip_adult_score =self ._get_clip_adult_score_only (image_features ,pil_image )
                    logger .info (f"[NSFW_HYBRID] NSFW={nsfw_score :.2f} düşük (<{nsfw_threshold }), CLIP'e soruldu: {clip_adult_score :.2f}")
                else :
                # ERSIN NSFW yüksek skor verdi, CLIP'e sormaya gerek yok (performans kazancı)
                    clip_adult_score =0.0 
                    logger .info (f"[NSFW_FIRST] NSFW={nsfw_score :.2f} yüksek (>={nsfw_threshold }), CLIP'e sorulmadı (performans kazancı)")

                    # ERSIN adult_content'i categories listesinden çıkar (CLIP döngüsünde kullanılmayacak)
                categories =[cat for cat in categories if cat !='adult_content']
                logger .info (f"[CATEGORIES_AFTER_NSFW] NSFW sonrası kategoriler: {categories }")

            final_scores ={}
            logger .info (f"[FINAL_SCORES_INIT] final_scores başlatıldı, categories: {categories }")

            # ERSIN Eğitilmiş classification head varsa onu kullan, yoksa prompt-based yaklaşım
            if self .classification_head is not None :
                logger .info ("Eğitilmiş classification head ile analiz yapılıyor")
                with torch .no_grad ():
                # ERSIN Classification head ile tahmin
                    predictions =self .classification_head (image_features )
                    predictions =predictions .squeeze (0 ).cpu ().numpy ()

                    # ERSIN Debug: Predictions kontrolü
                    if len (predictions )!=len (categories ):
                        logger .error (f"[CLASSIFICATION_HEAD_ERROR] Predictions length ({len (predictions )}) != categories length ({len (categories )})")

                        # ERSIN Skorları kategorilere ata
                    for i ,cat in enumerate (categories ):
                        if i >=len (predictions ):
                            logger .warning (f"[CLASSIFICATION_HEAD] Index {i } out of range for category {cat }, using 0.0")
                            base_score =0.0 
                        else :
                            base_score =float (predictions [i ])

                            # ERSIN Ensemble düzeltmesi uygula
                        if self .ensemble_service :
                            try :
                            # ERSIN Kategori için açıklama oluştur
                                base_description =f"{cat } content detected"

                                # ERSIN Ensemble prediction
                                # ERSIN Convert content_id ve person_id from str to int if needed
                                content_id_int =int (content_id )if content_id is not None and isinstance (content_id ,str )and content_id .isdigit ()else (content_id if isinstance (content_id ,int )else None )
                                person_id_int =int (person_id )if person_id is not None and isinstance (person_id ,str )and person_id .isdigit ()else (person_id if isinstance (person_id ,int )else None )
                                _ ,corrected_confidence ,correction_info =self .ensemble_service .clip_ensemble .predict_content_ensemble (
                                base_description =base_description ,
                                base_confidence =base_score ,
                                content_id =content_id_int ,
                                person_id =person_id_int ,
                                clip_embedding =image_features .cpu ().numpy ().squeeze ()
                                )

                                # ERSIN Düzeltilmiş skoru kullan
                                final_scores [cat ]=corrected_confidence 

                                if correction_info .get ('method')!='base_model':
                                    logger .info (f"[ENSEMBLE_CORRECTION] {cat }: {base_score :.3f} -> {corrected_confidence :.3f} ({correction_info .get ('method')})")

                            except Exception as e :
                                logger .warning (f"Ensemble correction failed for {cat }: {str (e )}")
                                final_scores [cat ]=base_score 
                        else :
                            final_scores [cat ]=base_score 

                    logger .info (f"[TRAINED_MODEL_LOG] Predictions: {dict (zip (categories ,predictions ))}")
            else :
                logger .info (f"Prompt-based analiz yapılıyor. Kategoriler: {categories }, Sayı: {len (categories )}")
                if len (categories )==0 :
                    logger .error ("[CRITICAL] categories listesi BOŞ! CLIP döngüsü çalışmayacak!")
                    # ERSIN Orijinal prompt-based yaklaşım
                for cat in categories :
                    logger .debug (f"[CLIP_LOOP] Kategori işleniyor: {cat }")
                    # ERSIN Template ensembling (ağırlıklandırma yok): her concept için 2–4 template üret
                    pos_prompts =[
                    tpl .format (x =p )
                    for p in self .category_prompts [cat ]["positive"]
                    for tpl in self .prompt_templates 
                    ]
                    neg_prompts =[
                    tpl .format (x =p )
                    for p in self .category_prompts [cat ]["negative"]
                    for tpl in self .prompt_templates 
                    ]
                    all_prompts =pos_prompts +neg_prompts 
                    device =getattr (self ,"device",("cuda"if torch .cuda .is_available ()else "cpu"))
                    text_inputs =self .tokenizer (all_prompts ).to (device )
                    with torch .no_grad ():
                    # ERSIN OpenCLIP encode_text() exists but type checker doesn't recognize it
                        encode_text_method =getattr (self .clip_model ,'encode_text',None )
                        if encode_text_method is not None and callable (encode_text_method ):
                            text_features =encode_text_method (text_inputs )
                        else :
                            raise RuntimeError ("CLIP model encode_text method not available")
                        # ERSIN Type checker için text_features'ı torch.Tensor olarak cast et
                        if text_features is not None :
                            text_features_tensor =cast (torch .Tensor ,text_features )
                            text_features =text_features_tensor /text_features_tensor .norm (dim =-1 ,keepdim =True )
                        if text_features is not None and image_features is not None :
                            text_features_t =cast (torch .Tensor ,text_features )
                            image_features_t =cast (torch .Tensor ,image_features )
                            similarities =(image_features_t @text_features_t .T ).squeeze (0 ).cpu ().numpy ()# ERSIN len = len(all_prompts)
                        else :
                            similarities =np .array ([])

                        # ERSIN Debug: similarities array kontrolü
                    if len (similarities )==0 :
                        logger .warning (f"[CLIP_DEBUG] {cat }: similarities array boş! CLIP çalışmıyor olabilir.")
                        final_scores [cat ]=0.0 
                        continue 

                    pos_sims =similarities [:len (pos_prompts )]
                    # ERSIN adult_content için pozitif tarafta ortalama çok "yumuşak" kalabiliyor.
                    # ERSIN Bariz karelerde en güçlü pozitif eşleşmeleri öne çıkarmak için p90 kullan.
                    if cat =="adult_content"and len (pos_sims )>0 :
                        pos_score =float (np .percentile (pos_sims ,90 ))
                    else :
                        pos_score =float (np .mean (pos_sims ))if len (pos_sims )>0 else 0.0 

                    neg_sims =similarities [len (pos_prompts ):]
                    # ERSIN adult_content için "hard negative" (max) tek bir benign-negative prompt'un
                    # ERSIN güçlü eşleşmesinde skoru aşırı bastırabiliyor. Bunun yerine p90 kullanarak
                    # ERSIN daha yumuşak ama hâlâ temkinli bir negatif agregasyon uygula.
                    if cat =="adult_content"and len (neg_sims )>0 :
                        neg_score =float (np .percentile (neg_sims ,90 ))
                    else :
                        neg_score =float (np .mean (neg_sims ))if len (neg_sims )>0 else 0.0 
                    fark =pos_score -neg_score 

                    # ERSIN Score calculation (prompt-based):
                    # ERSIN We already map the pos-neg difference into [0..1] via tanh.
                    # ERSIN The previous "MIN/MAX normalization" band (0.42..0.58) caused extreme saturation
                    # ERSIN (many frames becoming 1.0), making prompt comparisons meaningless.
                    import math 

                    # ERSIN Category-specific squashing:
                    # ERSIN adult_content prompt farkları genelde küçük (CLIP sim aralıkları dar),
                    # ERSIN bu yüzden daha agresif squashing kullanarak bariz içerikleri yukarı taşı.
                    SQUASH_FACTOR =18.0 if cat =="adult_content"else 6.0 
                    squashed_fark =math .tanh (fark *SQUASH_FACTOR )
                    # ERSIN Önemli: For "risk" categories, neutral evidence (pos≈neg) must map to ~0,
                    # ERSIN not 0.5. Otherwise benign content (e.g. baby photos) gets 30–50% "risk"
                    # ERSIN bars just from noise. We clamp negative evidence to 0 ve keep only
                    # ERSIN positive evidence.
                    base_score =max (0.0 ,min (1.0 ,float (max (0.0 ,squashed_fark ))))

                    # ERSIN Ham CLIP skorlarını logla (her frame için tüm kategoriler)
                    frame_name =os .path .basename (image_path )if isinstance (image_path ,str )else "numpy_array"
                    logger .info (
                    f"[CLIP_HAM_SKOR] frame={frame_name } kategori={cat } | "
                    f"pos_score={pos_score :.4f} neg_score={neg_score :.4f} fark={fark :.4f} | "
                    f"squashed={squashed_fark :.4f} base_score={base_score :.4f} | "
                    f"pos_sims_min={float (np .min (pos_sims )):.4f} pos_sims_max={float (np .max (pos_sims )):.4f} pos_sims_mean={float (np .mean (pos_sims )):.4f} | "
                    f"neg_sims_min={float (np .min (neg_sims )):.4f} neg_sims_max={float (np .max (neg_sims )):.4f} neg_sims_mean={float (np .mean (neg_sims )):.4f} | "
                    f"n_pos={len (pos_prompts )} n_neg={len (neg_sims )}"
                    )

                    # ERSIN Debug: Tüm kategoriler için skor logla (ilk birkaç kategori için)
                    if cat in ['violence','adult_content']or base_score >0.1 :
                        logger .debug (f"[CLIP_SCORE] {cat }: pos={pos_score :.4f}, neg={neg_score :.4f}, fark={fark :.4f}, squashed={squashed_fark :.4f}, base={base_score :.4f}")

                    if cat =="adult_content":
                    # ERSIN Prompt/score debug: CLIP aslında "adult" positive'leri görüyor mu, yoksa
                    # ERSIN negatif promptlar mı bastırıyor hızlıca anlamak için.
                        if base_score <0.15 or base_score >0.6 :
                            logger .info (
                            "[CLIP_ADULT_PROMPTS] pos=%.4f neg(p90)=%.4f fark=%.4f base=%.4f n_pos=%d n_neg=%d",
                            pos_score ,
                            neg_score ,
                            fark ,
                            base_score ,
                            len (pos_prompts ),
                            len (neg_sims ),
                            )

                            # ERSIN Ensemble düzeltmesi uygula
                    if self .ensemble_service :
                        try :
                            base_description =f"{cat } content detected"

                            # ERSIN Convert content_id ve person_id from str to int if needed
                            content_id_int =int (content_id )if content_id is not None and isinstance (content_id ,str )and content_id .isdigit ()else (content_id if isinstance (content_id ,int )else None )
                            person_id_int =int (person_id )if person_id is not None and isinstance (person_id ,str )and person_id .isdigit ()else (person_id if isinstance (person_id ,int )else None )
                            _ ,corrected_confidence ,correction_info =self .ensemble_service .clip_ensemble .predict_content_ensemble (
                            base_description =base_description ,
                            base_confidence =base_score ,
                            content_id =content_id_int ,
                            person_id =person_id_int ,
                            clip_embedding =image_features .cpu ().numpy ().squeeze ()
                            )

                            final_scores [cat ]=corrected_confidence 

                            if correction_info .get ('method')!='base_model':
                                logger .info (f"[ENSEMBLE_CORRECTION] {cat }: {base_score :.3f} -> {corrected_confidence :.3f} ({correction_info .get ('method')})")

                        except Exception as e :
                            logger .warning (f"Ensemble correction failed for {cat }: {str (e )}")
                            final_scores [cat ]=base_score 
                    else :
                        final_scores [cat ]=base_score 

                        # ERSIN adult_content skorunu NSFW ve CLIP'ten al (hibrit yaklaşım)
            if nsfw_enabled :
            # ERSIN Hibrit yaklaşım: Hangisi yüksekse onu kullan (max)
            # ERSIN NSFW bulduysa direkt NSFW, bulamadıysa CLIP'e sorduk ve max alıyoruz
                if clip_adult_score >0.0 :
                # ERSIN CLIP'e soruldu (NSFW düşüktü), hangisi yüksekse onu kullan
                    final_scores ['adult_content']=max (nsfw_score ,clip_adult_score )
                    logger .info (f"[NSFW_HYBRID] NSFW={nsfw_score :.2f}, CLIP={clip_adult_score :.2f} → max kullanıldı: {final_scores ['adult_content']:.2f}")
                else :
                # ERSIN NSFW yüksekti, CLIP'e sorulmadı, direkt NSFW kullan
                    final_scores ['adult_content']=nsfw_score 
                    logger .info (f"[NSFW_FIRST] adult_content={nsfw_score :.2f} (NSFW direkt, CLIP'e sorulmadı)")

                    # ERSIN "safe" skorunu diğer risklerden türet
                    # ERSIN UX PRINCIPLE: Eğer herhangi bir risk kategorisi yüksekse, safe skoru düşük olmalı
                    # ERSIN Örnek: adult_content=0.93 ise safe kesinlikle <0.2 olmalı
            risk_categories_for_safe_calculation =['violence','adult_content','harassment','weapon','drug']

            # ERSIN Maksimum risk skorunu bul (en yüksek risk kategori)
            max_risk_score =max (final_scores .get (rc ,0 )for rc in risk_categories_for_safe_calculation )if risk_categories_for_safe_calculation else 0 

            # ERSIN UX: Maksimum risk skorunu direkt kullan - eğer bir kategori yüksekse safe düşük olmalı
            # ERSIN Örnek: adult_content=0.93 → safe=0.07 (mantıklı)
            # ERSIN Önceki yöntem: adult_content=0.93 → safe=0.78 (mantıksız!)
            final_scores ['safe']=max (0.0 ,1.0 -max_risk_score )

            # ERSIN Eğer maksimum risk çok yüksekse (>0.8), safe skorunu daha da düşür
            # ERSIN Bu, kullanıcıya net bir mesaj verir: yüksek risk = düşük güvenlik
            if max_risk_score >0.8 :
            # ERSIN Çok yüksek risk durumunda safe skorunu maksimum risk'in karesi ile hesapla
            # ERSIN Örnek: max_risk=0.93 → safe = 1 - (0.93^1.2) = 1 - 0.92 = 0.08
                final_scores ['safe']=max (0.0 ,1.0 -(max_risk_score **1.2 ))

                # ERSIN Debug: Tüm skorları logla
            logger .info (f"[SCORE_SUMMARY] Final scores: {dict (final_scores )}")
            logger .info (f"[SAFE_SCORE_CALC] Max risk: {max_risk_score :.4f}, Calculated safe score: {final_scores ['safe']:.4f}")

            # ERSIN Tespit edilen nesneleri Python tiplerine dönüştür
            safe_objects =convert_numpy_types_to_python (detected_objects )

            # ERSIN Bağlamsal ayarlamalar (sanity-mode'da kapalı)
            person_count =len ([obj for obj in safe_objects if obj .get ('label')=='person'])
            object_labels =[obj .get ('label')for obj in safe_objects if obj .get ('label')]

            # ERSIN ERSIN _apply_contextual_adjustments çağrılmadan önce tüm kategorilerin (safe dah...
            # ERSIN 'categories' listesi artık self.category_prompts.keys() ile aynı (safe hariç)
            # ERSIN Ancak _apply_contextual_adjustments 'safe' dahil tüm kategorileri bekliyor olabilir.
            # ERSIN Dönüş değerinde de tüm kategoriler bekleniyor.
            # ERSIN all_category_keys_for_return şu an kullanılmıyor
            _ =list (self .category_prompts .keys ())+['safe']# ERSIN safe'i manuel ekle

            if not getattr (self ,"sanity_mode",False ):
                final_scores =self ._apply_contextual_adjustments (final_scores ,object_labels ,person_count )

                # ERSIN Düzenlenen skorları döndür
                # ERSIN Return sırası: violence, adult_content, harassment, weapon, drug, safe
                # ERSIN analysis_service.py'de bu sırada unpack ediliyor
            return (
            final_scores .get ('violence',0.0 ),
            final_scores .get ('adult_content',0.0 ),
            final_scores .get ('harassment',0.0 ),
            final_scores .get ('weapon',0.0 ),
            final_scores .get ('drug',0.0 ),
            final_scores .get ('safe',0.0 ),
            safe_objects 
            )
        except Exception as e :
            logger .error (f"CLIP görüntü analizi hatası: {str (e )}")
            raise 

    def _apply_contextual_adjustments (self ,scores :dict [str ,float ],object_labels :list [str ],person_count :int )->dict [str ,float ]:
        """
        Nesne tespitine dayalı bağlamsal ayarlamalar yapar.
        Bu fonksiyon, CLIP sonuçlarını tespit edilen nesnelere göre ayarlar.
        """
        logger .info (f"[ContextualAdjust] Fonksiyon başlangıcı. Gelen skorlar: {scores }, Nesne etiketleri: {object_labels }, Kişi sayısı: {person_count }")

        # ERSIN Silahla ilgili nesneler ve potansiyel riskli nesneler
        weapon_objects =['gun','rifle','pistol','shotgun','weapon','explosive','bomb']
        drug_objects =['bottle','wine glass','cup','cigarette','syringe','pipe','bong','pills','powder']# ERSIN 'bottle', 'cup' gibi genel nesneler dikkatli kullanılmalı

        # ERSIN Mutfakla ilgili nesneler (yanlış pozitifleri azaltmak için)
        kitchen_objects =['oven','refrigerator','sink','microwave','kitchen','restaurant','dining table','food','plate']

        # ERSIN Riskli nesne tespiti yapıldı mı?
        weapon_detected =any (obj in object_labels for obj in weapon_objects )
        drug_related_detected =any (obj in object_labels for obj in drug_objects if obj not in ['bottle','cup'])# ERSIN Daha spesifik uyuşturucu nesneleri
        general_drug_indicators =any (obj in ['bottle','cup']for obj in object_labels )# ERSIN Şişe, bardak gibi genel ama bağlama göre riskli olabilecekler

        logger .info (f"[ContextualAdjust] Hesaplanan bayraklar -> weapon_detected: {weapon_detected }, drug_related_detected: {drug_related_detected }, general_drug_indicators: {general_drug_indicators }")

        if weapon_detected :
            is_kitchen_context_with_knife ='knife'in object_labels and any (ko in object_labels for ko in kitchen_objects )
            if is_kitchen_context_with_knife :
            # ERSIN Mutfak bıçağı daha az riskli kabul edilebilir
                scores ['weapon']=min (scores ['weapon']*1.1 ,1.0 )# ERSIN Hafif artış
                scores ['violence']=min (scores ['violence']*1.1 ,1.0 )
                scores ['safe']=max (scores ['safe']*0.8 ,0.0 )
                logger .info ("Mutfak bağlamında bıçak tespit edildi, silah/şiddet skorları daha az artırıldı.")
            else :
            # ERSIN Genel silah tespiti
                scores ['weapon']=min (scores ['weapon']*1.5 ,1.0 )
                scores ['violence']=min (scores ['violence']*1.3 ,1.0 )
                scores ['safe']=max (scores ['safe']*0.5 ,0.0 )
                logger .info ("Tespit edilen silah nesneleri, silah/şiddet skorları artırıldı")

        if drug_related_detected :
            scores ['drug']=min (scores ['drug']*1.4 ,1.0 )# ERSIN Spesifik uyuşturucu nesneleri için daha güçlü artış
            scores ['safe']=max (scores ['safe']*0.6 ,0.0 )
            logger .info ("Tespit edilen spesifik madde kullanımı göstergeleri, madde skoru artırıldı")
        elif general_drug_indicators and person_count >0 :# ERSIN Şişe, bardak gibi nesneler ve insanlar varsa
        # ERSIN ERSIN Bu durum daha belirsiz olduğu için 'drug' skorunu daha az etkileyebilir ve...
        # ERSIN Örneğin, 'party' (parti) gibi bir nesne tespitiyle birleştirilebilir
            if 'party'in object_labels or 'bar'in object_labels :
                 scores ['drug']=min (scores ['drug']*1.1 ,1.0 )
                 logger .info ("Genel madde kullanımı göstergeleri (şişe/bardak) parti/bar bağlamında, madde skoru hafif artırıldı")

                 # ERSIN GÜVENLİ KATEGORİSİ İÇİN GÜÇLENDİRME
                 # ERSIN UX PRINCIPLE: Eğer herhangi bir yüksek risk varsa, safe skorunu ASLA artırma
                 # ERSIN Önce yüksek risk kontrolü yap
        other_categories_for_safe_check =['violence','adult_content','harassment','weapon','drug']
        high_risk_threshold =0.5 # ERSIN Eğer herhangi bir kategori >0.5 ise, safe artırma
        any_high_risk_score =any (scores .get (cat ,0 )>high_risk_threshold for cat in other_categories_for_safe_check )

        # ERSIN Eğer yüksek risk varsa, safe skorunu artırma - zaten düşük olmalı
        if any_high_risk_score :
            logger .info (f"[ContextualAdjust] Yüksek risk tespit edildi, safe skoru artırılmayacak. Max risk: {max (scores .get (cat ,0 )for cat in other_categories_for_safe_check ):.2f}")
            return scores # ERSIN Safe skorunu olduğu gibi bırak

            # ERSIN Sadece tüm riskler düşükse safe skorunu artırabiliriz
        no_immediate_risk_objects =not weapon_detected and not drug_related_detected 
        low_score_threshold =0.3 # ERSIN Düşük risk eşiği (daha sıkı)
        all_other_scores_truly_low =all (scores .get (cat ,0 )<low_score_threshold for cat in other_categories_for_safe_check )
        most_scores_medium_or_low =sum (1 for cat in other_categories_for_safe_check if scores .get (cat ,0 )<0.4 )>=4 

        logger .info (f"[ContextualAdjust] Güvenli kategori analizi -> all_low: {all_other_scores_truly_low }, most_medium_low: {most_scores_medium_or_low }, any_high: {any_high_risk_score }")

        if no_immediate_risk_objects and all_other_scores_truly_low and person_count <=2 :
        # ERSIN Çok güvenli durum - safe skorunu yüksek seviyeye çıkar
            original_safe_score =scores .get ('safe',0 )
            scores ['safe']=min (0.9 ,scores .get ('safe',0 )*1.1 )# ERSIN Maksimum %10 artır
            logger .info (f"ÇOK GÜVENLİ: 'safe' skoru {original_safe_score :.2f} -> {scores ['safe']:.2f}")

        elif no_immediate_risk_objects and most_scores_medium_or_low and person_count <=4 :
        # ERSIN Orta güvenli durum - safe skorunu hafifçe artır
            original_safe_score =scores .get ('safe',0 )
            scores ['safe']=min (0.8 ,scores .get ('safe',0 )*1.05 )# ERSIN Maksimum %5 artır
            logger .info (f"ORTA GÜVENLİ: 'safe' skoru {original_safe_score :.2f} -> {scores ['safe']:.2f}")

            # ERSIN YÜKSEK RİSK SKORLARINI YOLO ONAYI OLMADAN DÜŞÜRME
        high_clip_threshold =0.6 # ERSIN Belirsiz seviyenin üstü
        logger .info (f"[ContextualAdjust] Yüksek CLIP skoru düşürme parametreleri -> threshold: {high_clip_threshold }")

        # ERSIN Silah için geliştirilmiş kontrol (YOLO onayı yoksa daha agresif düşürme)
        if scores .get ('weapon',0 )>high_clip_threshold and not weapon_detected :
            original_weapon_score =scores ['weapon']
            # ERSIN Mutfak bıçağı kontrolü - çarpan kullan, sabit değer değil
            if 'knife'in object_labels and any (ko in object_labels for ko in ['oven','refrigerator','sink','microwave','kitchen']):
                scores ['weapon']=scores ['weapon']*0.3 # ERSIN Mutfakta daha agresif risk azaltma
                logger .info (f"Mutfak bağlamında silah skoru düşürülüyor: {original_weapon_score :.2f} → {scores ['weapon']:.2f}")
                # ERSIN Telefon/cüzdan gibi nesneler için kontrol
            elif any (obj in object_labels for obj in ['cell phone','phone','mobile phone','smartphone']):
                scores ['weapon']=scores ['weapon']*0.3 # ERSIN Telefon için çok agresif düşürme
                logger .info (f"Telefon tespit edildi, silah skoru agresif düşürülüyor: {original_weapon_score :.2f} → {scores ['weapon']:.2f}")
            else :
                scores ['weapon']=scores ['weapon']*0.4 # ERSIN Genel risk azaltma (0.7'den 0.4'e düşürüldü - daha agresif)

            logger .info (f"CLIP 'weapon' skoru yüksek ({original_weapon_score :.2f}) ama YOLO onayı yok, skor {scores ['weapon']:.2f}'ye düşürüldü")

            # ERSIN Madde için geliştirilmiş kontrol
        if scores .get ('drug',0 )>high_clip_threshold and not drug_related_detected :
            original_drug_score =scores ['drug']
            scores ['drug']=scores ['drug']*0.7 # ERSIN Risk azaltma, sabit değer değil
            logger .info (f"CLIP 'drug' skoru yüksek ({original_drug_score :.2f}) ama YOLO spesifik onayı yok, skor {scores ['drug']:.2f}'ye düşürüldü")

        return scores 

        # ERSIN Backward compatibility: eski fonksiyon isimleri için alias
    analyze_content =analyze_image 
    _preprocess_image =lambda self ,*args ,**kwargs :self .clip_preprocess (*args ,**kwargs )if hasattr (self ,'clip_preprocess')else None 
    _get_text_features =lambda self ,*args ,**kwargs :self .clip_model .encode_text (*args ,**kwargs )if hasattr (self ,'clip_model')else None 
    _calculate_similarities =lambda self ,img_feat ,txt_feat :(img_feat @txt_feat .T ).cpu ().numpy ()if hasattr (self ,'clip_model')else None 

def get_content_analyzer ()->ContentAnalyzer :
    """
    Performance-optimized factory function for ContentAnalyzer singleton
    
    Returns:
        ContentAnalyzer: Thread-safe singleton instance
    """
    return ContentAnalyzer ()