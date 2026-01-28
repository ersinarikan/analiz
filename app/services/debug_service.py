"""
Debug servisi
Bu modül, analiz işlemlerinin durumunu takip etmek ve sorunları tespit etmek için kullanılır.
"""
import logging 
from app .models .analysis import Analysis 
from app import db 
from datetime import datetime ,timedelta 
import json 
import numpy as np 
import traceback 

logger =logging .getLogger (__name__ )

def log_active_analyses ():
    """
    Aktif analiz işlemlerinin durumunu loglar.
    Bu fonksiyon, sistemdeki bekleyen ve işlem gören analizleri kontrol eder
    ve uzun süre aynı durumda kalan analizleri tespit eder.
    """
    try :
    # ERSIN ERSIN Son 1 saat içinde oluşturulmuş ve hala 'pending' veya 'processing' durumun...
        cutoff_time =datetime .utcnow ()-timedelta (hours =1 )
        from app .utils .sqlalchemy_helpers import column_in 
        from sqlalchemy import and_ 
        from sqlalchemy .sql import ColumnElement 
        from typing import cast ,Any
        # ERSIN column_in returns a filter expression compatible with SQLAlchemy
        status_filter =column_in (Analysis .status ,['pending','processing'])
        # ERSIN Type checker için created_at'i cast et
        created_at_col =cast (ColumnElement [Any ],Analysis .created_at )
        active_analyses =Analysis .query .filter (
        and_ (
        status_filter ,
        created_at_col >=cutoff_time 
        )
        ).all ()

        if active_analyses :
            logger .info (f"Aktif analiz sayısı: {len (active_analyses )}")

            for analysis in active_analyses :
            # ERSIN Başlangıçtan beri geçen süreyi hesapla
                elapsed_time =datetime .utcnow ()-analysis .created_at 
                elapsed_minutes =elapsed_time .total_seconds ()/60 

                # ERSIN Uzun süre aynı durumda kalan analizleri logla
                if elapsed_minutes >5 :# ERSIN 5 dakikadan uzun süre aynı durumda kalan analizler
                    logger .warning (
                    f"Analiz #{analysis .id } {elapsed_minutes :.1f} dakikadır '{analysis .status }' durumunda. "
                    f"Dosya: #{analysis .file_id } - WebSocket üzerinden takip edilebilir"
                    )

                    # ERSIN Eğer analiz 30 dakikadan uzun süredir aynı durumda kaldıysa otomatik olara...
                    if elapsed_minutes >30 and analysis .status =='processing':
                        logger .error (f"Analiz #{analysis .id } zaman aşımına uğradı (30+ dakika). Başarısız olarak işaretleniyor.")
                        analysis .fail_analysis ("Analiz işlemi zaman aşımına uğradı")
                        db .session .commit ()
                else :
                    logger .debug (
                    f"Analiz #{analysis .id } {elapsed_minutes :.1f} dakikadır '{analysis .status }' durumunda."
                    )
        else :
            logger .debug ("Aktif analiz işlemi yok")

    except Exception as e :
        logger .error (f"Aktif analizleri loglarken hata: {str (e )}")

def repair_stuck_analyses ():
    """
    Takılmış analiz işlemlerini tespit edip düzeltmeyi dener.
    Bu fonksiyon, uzun süre 'processing' durumunda kalan analizleri bulur 
    ve bunları 'failed' durumuna getirir, böylece kullanıcı tekrar deneyebilir.
    """
    try :
    # ERSIN 15 dakikadan uzun süredir 'processing' durumunda olan analizleri al
        cutoff_time =datetime .utcnow ()-timedelta (minutes =15 )
        stuck_analyses =Analysis .query .filter (
        (Analysis .status =='processing')&
        (Analysis .updated_at <=cutoff_time )
        ).all ()

        if stuck_analyses :
            logger .warning (f"{len (stuck_analyses )} takılmış analiz tespit edildi. Düzeltiliyor...")

            for analysis in stuck_analyses :
                elapsed_time =datetime .utcnow ()-analysis .updated_at 
                elapsed_minutes =elapsed_time .total_seconds ()/60 

                logger .warning (
                f"Takılmış analiz #{analysis .id } tespit edildi. "
                f"Son güncelleme: {elapsed_minutes :.1f} dakika önce"
                )

                # ERSIN Analizi başarısız olarak işaretle
                analysis .fail_analysis ("Analiz işlemi yanıt vermeyi durdurdu ve otomatik olarak iptal edildi")

            db .session .commit ()
            logger .info (f"{len (stuck_analyses )} takılmış analiz başarısız olarak işaretlendi")

    except Exception as e :
        logger .error (f"Takılmış analizleri düzeltirken hata: {str (e )}")

def test_numpy_serialization ():
    """NumPy veri türlerinin JSON serileştirme işlemini test eder."""
    try :
    # ERSIN Test nesneleri oluştur
        test_objects =[
        {
        'label':'person',
        'confidence':np .float32 (0.95 ),
        'box':[np .int32 (10 ),np .int32 (20 ),np .int32 (30 ),np .int32 (40 )]
        },
        {
        'label':'car',
        'confidence':np .float64 (0.85 ),
        'box':np .array ([50 ,60 ,70 ,80 ])
        }
        ]

        # ERSIN NumPy veri dönüşümü işlevi
        def convert_numpy_types (obj ):
            if isinstance (obj ,np .integer ):
                return int (obj )
            elif isinstance (obj ,np .floating ):
                return float (obj )
            elif isinstance (obj ,np .ndarray ):
                return convert_numpy_types (obj .tolist ())
            elif isinstance (obj ,(list ,tuple )):
                return [convert_numpy_types (item )for item in obj ]
            elif isinstance (obj ,dict ):
                return {key :convert_numpy_types (value )for key ,value in obj .items ()}
            else :
                return obj 

                # ERSIN Test 1: Dönüşümsüz doğrudan serileştirme dene
        try :
            json_string_direct =json .dumps (test_objects )
            logger .info ("Doğrudan serileştirme başarılı. Sonuç: "+json_string_direct [:100 ]+"...")
        except TypeError as e :
            logger .error (f"Doğrudan serileştirme hatası: {e }")

            # ERSIN Test 2: Dönüşüm işlevi ile serileştirme dene
        safe_objects =convert_numpy_types (test_objects )
        try :
            json_string_converted =json .dumps (safe_objects )
            logger .info ("Dönüştürülmüş serileştirme başarılı. Sonuç: "+json_string_converted [:100 ]+"...")
            return True ,json_string_converted 
        except TypeError as e :
            logger .error (f"Dönüştürülmüş serileştirme hatası: {e }")
            return False ,str (e )
    except Exception as e :
        logger .error (f"Test sırasında beklenmeyen hata: {e }")
        logger .error (traceback .format_exc ())
        return False ,str (e )

def debug_object (obj ,name ="Object"):
    """Bir nesnenin detaylı hata ayıklama bilgisini loglar."""
    try :
        logger .info (f"DEBUG {name } type: {type (obj )}")
        if isinstance (obj ,dict ):
            for key ,value in obj .items ():
                logger .info (f"DEBUG {name }[{key }] = {type (value )}: {value }")
        elif isinstance (obj ,(list ,tuple )):
            for i ,item in enumerate (obj ):
                logger .info (f"DEBUG {name }[{i }] = {type (item )}: {str (item )[:100 ]}")
        else :
            logger .info (f"DEBUG {name } value: {str (obj )[:200 ]}")

            # ERSIN Serileştirme testi
        try :
            json_str =json .dumps (obj )
            logger .info (f"DEBUG {name } is JSON serializable. Length: {len (json_str )}")
        except TypeError as e :
            logger .error (f"DEBUG {name } is NOT JSON serializable: {e }")
    except Exception as e :
        logger .error (f"Debug logging error for {name }: {e }")