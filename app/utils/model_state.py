# ERSIN Model State File
# ERSIN Bu dosya model aktivasyonlarında güncellenir ve Flask debug mode tarafından izlenir
# ERSIN Otomatik restart için config.py tarafından import edilir
# pyright: reportConstantRedefinition=false
# ERSIN Mutable global değişkenler uppercase kullanıyor (backward compatibility), type checker'a mutable olduğunu söyle

import threading 
import time 
from datetime import datetime 
import logging 
from typing import Any 

logger =logging .getLogger (__name__ )

# ERSIN thread-safe model state management
_state_lock =threading .Lock ()

# ERSIN Model instance cache için performance optimization
_model_instances ={}
_model_lock =threading .Lock ()

# ERSIN Type checker için MODEL_STATE'i doğru tiplerle initialize et
# ERSIN Mutable global değişkenler için lowercase kullan (type checker constant algılamasını önler)
_model_state :dict [str ,dict [str ,Any ]]={
'age':{
'active_version':1 ,# ERSIN En son versiyon aktif
'last_activation':datetime .now ().isoformat ()
},
'content':{
'active_version':None ,
'last_activation':None 
}
}

# ERSIN Public API - uppercase alias'lar için getter/setter pattern
def _get_model_state ()->dict [str ,dict [str ,Any ]]:
    return _model_state 

def _set_model_state (value :dict [str ,dict [str ,Any ]])->None :
    global _model_state 
    _model_state =value 

# ERSIN Backward compatibility için uppercase module-level variable (runtime'da mutable)
# ERSIN Type checker için @property kullanmıyoruz, direkt dict referansı kullanıyoruz
MODEL_STATE =_model_state  # ERSIN Mutable global, backward compatibility için uppercase alias

# ERSIN Bu satır Flask'ın dosya değişikliklerini algılaması için
# ERSIN Her model aktivasyonunda timestamp güncellenir
# ERSIN Type checker için mutable olarak işaretle (lowercase kullan)
_last_update :str ="2025-05-30T18:54:00.000000"

def _get_last_update ()->str :
    return _last_update 

def _set_last_update (value :str )->None :
    global _last_update 
    _last_update =value 

# ERSIN Backward compatibility için uppercase module-level variable
LAST_UPDATE =_last_update  # ERSIN Mutable global, backward compatibility için uppercase alias

def update_model_state (model_type :str ,version_id )->None :
    """
    Thread-safe model state güncelleme.
    Args:
        model_type (str): Model tipi ('age', 'content', etc.).
        version_id: Versiyon ID'si (int, str, or None).
    Returns:
        None
    """
    # ERSIN Type checker için global değişkenleri güncelle
    global MODEL_STATE ,LAST_UPDATE ,_model_state ,_last_update 
    from typing import cast

    with _state_lock :
        # ERSIN Type checker için mutable global değişkenleri kullan
        if model_type not in _model_state :
            _model_state [model_type ]={}

        _model_state [model_type ]['active_version']=version_id 
        _model_state [model_type ]['last_activation']=datetime .now ().isoformat ()
        # ERSIN Uppercase alias'ı güncelle (aynı dict referansı)
        MODEL_STATE =_model_state  # ERSIN Uppercase alias güncelle

        # ERSIN Flask dosya değişiklik algılaması için timestamp güncelle
        _last_update =datetime .now ().isoformat ()
        LAST_UPDATE =_last_update  # ERSIN Uppercase alias güncelle

def get_model_state (model_type =None ):
    """
    Thread-safe model state okuma
    
    Args:
        model_type (str, optional): Belirli model tipi, None ise tümü
        
    Returns:
        dict: Model state bilgisi
    """
    with _state_lock :
        if model_type :
            return _model_state .get (model_type ,{}).copy ()
        return _model_state .copy ()

def reset_model_state ():
    """
    Tüm model state'ini sıfırla
    """
    # ERSIN Type checker için global değişkenleri güncelle
    global MODEL_STATE ,LAST_UPDATE ,_model_state ,_last_update 
    from typing import cast

    with _state_lock :
        # ERSIN Type checker için MODEL_STATE'i doğru tiplerle reset et
        _model_state ={
        'age':{'active_version':0 ,'last_activation':None },
        'content':{'active_version':None ,'last_activation':None }
        }
        MODEL_STATE =_model_state  # ERSIN Uppercase alias güncelle
        _last_update =datetime .now ().isoformat ()
        LAST_UPDATE =_last_update  # ERSIN Uppercase alias güncelle

def get_age_estimator ():
    """
    Performance-optimized thread-safe singleton InsightFaceAgeEstimator
    
    Returns:
        InsightFaceAgeEstimator: Cached model instance
    """
    cache_key ='age_estimator'

    # ERSIN thread-safe cache kontrolü
    with _model_lock :
        if cache_key in _model_instances :
            instance =_model_instances [cache_key ]
            if instance is not None :
                logger .debug ("Age estimator cache'den kullanılıyor")
                return instance 

                # ERSIN Cache miss - yeni instance oluştur
    try :
        from app .ai .insightface_age_estimator import InsightFaceAgeEstimator 
        logger .info ("Yeni InsightFaceAgeEstimator instance oluşturuluyor...")

        start_time =time .time ()
        estimator =InsightFaceAgeEstimator ()# ERSIN CLIP shared olarak inject edilecek
        load_time =time .time ()-start_time 

        # ERSIN thread-safe cache'e kaydet
        with _model_lock :
            _model_instances [cache_key ]=estimator 

        logger .info (f"InsightFaceAgeEstimator cache'e kaydedildi ({load_time :.2f}s)")
        return estimator 

    except Exception as e :
        logger .error (f"InsightFaceAgeEstimator yükleme hatası: {e }")
        raise 

def get_content_analyzer ():
    """
    Performance-optimized thread-safe singleton ContentAnalyzer
    
    Returns:
        ContentAnalyzer: Cached model instance
    """
    cache_key ='content_analyzer'

    # ERSIN thread-safe cache kontrolü
    with _model_lock :
        if cache_key in _model_instances :
            instance =_model_instances [cache_key ]
            if instance is not None and hasattr (instance ,'initialized')and instance .initialized :
                logger .debug ("Content analyzer cache'den kullanılıyor")
                return instance 

                # ERSIN Cache miss - yeni instance oluştur
    try :
        from app .ai .content_analyzer import ContentAnalyzer 
        logger .info ("Yeni ContentAnalyzer instance oluşturuluyor...")

        start_time =time .time ()
        analyzer =ContentAnalyzer ()
        load_time =time .time ()-start_time 

        if analyzer .initialized :
        # ERSIN thread-safe cache'e kaydet
            with _model_lock :
                _model_instances [cache_key ]=analyzer 

            logger .info (f"ContentAnalyzer cache'e kaydedildi ({load_time :.2f}s)")
            return analyzer 
        else :
            raise RuntimeError ("ContentAnalyzer initialization failed")

    except Exception as e :
        logger .error (f"ContentAnalyzer yükleme hatası: {e }")
        raise 

def clear_model_cache (model_type =None ):
    """
    Model cache'ini temizle - memory management için
    
    Args:
        model_type (str, optional): Temizlenecek model tipi ('age', 'content'), None ise tümü
    """
    with _model_lock :
        if model_type =='age':
            if 'age_estimator'in _model_instances :
                instance =_model_instances ['age_estimator']
                if hasattr (instance ,'cleanup_models'):
                    instance .cleanup_models ()
                del _model_instances ['age_estimator']
                logger .info ("Age estimator cache temizlendi")

        elif model_type =='content':
            if 'content_analyzer'in _model_instances :
                instance =_model_instances ['content_analyzer']
                if hasattr (instance ,'cleanup_models'):
                    instance .cleanup_models ()
                del _model_instances ['content_analyzer']
                logger .info ("Content analyzer cache temizlendi")

        else :
        # ERSIN Tüm cache'i temizle
            for key ,instance in _model_instances .items ():
                if hasattr (instance ,'cleanup_models'):
                    instance .cleanup_models ()
            _model_instances .clear ()
            logger .info ("Tüm model cache temizlendi")

def get_cache_stats ():
    """
    Model cache istatistiklerini döndür
    
    Returns:
        dict: Cache istatistikleri
    """
    with _model_lock :
        # ERSIN Type checker için stats'ı doğru tiplerle initialize et
        stats :dict [str ,Any ]={
        'cached_models':list (_model_instances .keys ()),
        'cache_size':len (_model_instances ),
        'memory_usage':{}# ERSIN dict[str, dict[str, Any]]
        }

        for key ,instance in _model_instances .items ():
            # ERSIN Type checker için memory_usage'ı güvenli şekilde kullan
            memory_usage =stats .get ('memory_usage',{})
            if isinstance (memory_usage ,dict ):
                memory_usage [key ]={
                'type':type (instance ).__name__ ,
                'initialized':getattr (instance ,'initialized','N/A')
                }
                stats ['memory_usage']=memory_usage

        return stats 

def set_age_model_version (version_id ):
    """
    Yaş modeli aktif versiyonunu ayarla
    
    Args:
        version_id: Versiyon ID'si (int, 0 = base model)
    """
    update_model_state ('age',version_id )
    logger .info (f"Age model version set to: {version_id }")

def set_content_model_version (version_id ):
    """
    İçerik modeli aktif versiyonunu ayarla
    
    Args:
        version_id: Versiyon ID'si (int, 0 = base model) 
    """
    update_model_state ('content',version_id )
    logger .info (f"Content model version set to: {version_id }")

def update_model_state_file (model_type ,version_id ):
    """
    Model state dosyasını güncelle (backward compatibility için)
    
    Args:
        model_type (str): Model tipi ('age' veya 'content')
        version_id: Versiyon ID'si
    """
    update_model_state (model_type ,version_id )
    logger .info (f"Model state file updated: {model_type } -> version {version_id }")


def reset_model_cache ():
    """
    Model cache'ini temizler. Model aktivasyonundan sonra çağrılır.
    Bu sayede in-memory model instance'ları yeniden yüklenir.
    """
    global _model_instances 

    with _model_lock :
        old_count =len (_model_instances )
        _model_instances .clear ()
        logger .info (f"🔄 Model cache temizlendi ({old_count } instance kaldırıldı)")

        # ERSIN Force garbage collection için cached models
    import gc 
    gc .collect ()