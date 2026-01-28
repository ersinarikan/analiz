from datetime import datetime 
import json 
from app import db 
from app .models .file import File 
import logging 
from sqlalchemy import Column ,Integer ,String ,Float ,DateTime ,ForeignKey ,Text ,Boolean ,JSON ,func 
from sqlalchemy .sql import ColumnElement 
from app .json_encoder import NumPyJSONEncoder 
from sqlalchemy .orm import relationship 
import uuid 
from typing import TYPE_CHECKING ,Any 

if TYPE_CHECKING :
# ERSIN For type checking: SQLAlchemy columns are ColumnElement instances
    from sqlalchemy .sql import ColumnElement as ColumnType 
    from typing import Union 
    # ERSIN SQLAlchemy columns can be used as both class attributes (Column) and instance attributes (actual values)
    # ERSIN Use Union to represent both Column (class attribute) and actual value (instance attribute)
    StatusType =Union [ColumnType [Any ],str ]
    DateTimeType =Union [ColumnType [Any ],datetime ]
else :
    ColumnType =object 
    StatusType =object 
    DateTimeType =object 

logger =logging .getLogger (__name__ )

class Analysis (db .Model ):
    """
    Analiz ana modeli.
    - Yüklenen dosya, analiz türü, sonuçlar ve ilişkili içerik/yaş tespitleriyle bağlantı sağlar.
    """
    __tablename__ ='analyses'

    id =db .Column (db .String (36 ),primary_key =True ,default =lambda :str (uuid .uuid4 ()))
    file_id =db .Column (db .String (36 ),db .ForeignKey ('files.id'),nullable =False )
    # ERSIN SQLAlchemy columns can be used as both class attributes (Column) and instance attributes (actual values)
    # ERSIN Type annotation represents runtime behavior: Column at class level, actual value at instance level
    status :StatusType =db .Column (db .String (20 ),default ='pending')# ERSIN 'pending', 'processing', 'completed', 'failed'

    # ERSIN Use naive-UTC timestamps consistently across the app.
    start_time :DateTimeType =db .Column (db .DateTime ,default =datetime .utcnow )
    end_time :DateTimeType =db .Column (db .DateTime ,nullable =True )
    created_at :DateTimeType =db .Column (db .DateTime ,default =datetime .utcnow )

    frames_analyzed =db .Column (db .Integer ,default =0 )
    frames_per_second =db .Column (db .Float ,default =1.0 )

    error_message =db .Column (db .Text ,nullable =True )

    include_age_analysis =db .Column (db .Boolean ,default =False )# ERSIN Yaş tahmini yapılsın mı?

    # ERSIN WebSocket session tracking
    websocket_session_id =db .Column (db .String (255 ),nullable =True )# ERSIN WebSocket session ID
    is_cancelled =db .Column (db .Boolean ,default =False )# ERSIN Analiz iptal edildi mi?

    # ERSIN Genel kategorik skorlar (0-1 arası)
    overall_violence_score =db .Column (db .Float ,default =0 )
    overall_adult_content_score =db .Column (db .Float ,default =0 )
    overall_harassment_score =db .Column (db .Float ,default =0 )
    overall_weapon_score =db .Column (db .Float ,default =0 )
    overall_drug_score =db .Column (db .Float ,default =0 )
    overall_safe_score =db .Column (db .Float ,default =0 )# ERSIN Güvenli içerik skoru

    # ERSIN En yüksek riskli çerçeve bilgileri
    highest_risk_frame =db .Column (db .String (255 ))# ERSIN En riskli kare dosya yolu
    highest_risk_frame_timestamp =db .Column (db .Float )# ERSIN Kare zaman damgası (videolar için)
    highest_risk_score =db .Column (db .Float )# ERSIN En yüksek risk skoru
    highest_risk_category =db .Column (db .String (20 ))# ERSIN En riskli kategori ('violence', 'adult_content', vb.)

    # ERSIN Kategori bazlı en yüksek riskli çerçeve bilgileri (JSON olarak saklanacak)
    category_specific_highest_risks_data =db .Column (db .Text ,nullable =True )

    # ERSIN İlişkiler - Çakışmayı önlemek için file_ref ve file ilişkilerini kaldırdık
    # ERSIN Bunun yerine, tek bir file ilişkisi kullanacağız
    file =db .relationship ('File',
    foreign_keys =[file_id ],
    primaryjoin ="func.cast(File.id, String) == Analysis.file_id",
    back_populates ="analyses")
    content_detections =db .relationship ('ContentDetection',backref ='analysis',lazy =True ,cascade ="all, delete-orphan")
    age_estimations =db .relationship ('AgeEstimation',backref ='analysis',lazy =True ,cascade ="all, delete-orphan")

    def start_analysis (self ):
        """Analiz sürecini başlatır ve durumu 'processing' olarak günceller.

        NOTE: Model methods should not commit implicitly. Caller owns the transaction.
        """
        self .status ='processing'
        # ERSIN start_time default'u var, ama bazı path'lerde eski değer kalabiliyor; güvenli olmak için kontrol et
        # ERSIN Type checker sees start_time as DateTimeType, but at runtime it's datetime | None
        if self .start_time is None :
            self .start_time =datetime .utcnow ()

    def cancel_analysis (self ,reason ="WebSocket bağlantısı kesildi"):
        """Analizi iptal eder ve durumu günceller.

        NOTE: This method intentionally does NOT commit. Caller owns the transaction.
        """
        self .is_cancelled =True 
        self .status ='cancelled'
        self .error_message =reason 
        self .end_time =datetime .utcnow ()

    def check_if_cancelled (self ):
        """Analizin iptal edilip edilmediğini kontrol eder."""
        return self .is_cancelled 

    def update_progress (self ,progress :int ,message :str |None =None ):
        """
        Analiz ilerleme durumunu WebSocket ile bildirir.
        
        Args:
            progress: 0-100 arası ilerleme yüzdesi
            message: Opsiyonel durum mesajı
        """
        # ERSIN WebSocket ile progress bildirimi gönder
        logger .info (f"[DEBUG_PROGRESS] update_progress çağrıldı - Analysis: {self .id }, Progress: {progress }, Message: {message }")
        try :
            from app .routes .websocket_routes import emit_analysis_progress 
            result =emit_analysis_progress (
            analysis_id =self .id ,
            progress =min (progress ,100 ),
            message =message or f'İlerleme: %{min (progress ,100 )}',
            file_id =self .file_id 
            )
            logger .info (f"[DEBUG_PROGRESS] emit_analysis_progress sonucu: {result }")
        except Exception as ws_err :
            logger .warning (f"WebSocket progress event hatası: {str (ws_err )}")

    def complete_analysis (self ):
        """Analizi başarıyla tamamlandı olarak işaretler.

        NOTE: This method intentionally does NOT commit. Caller owns the transaction.
        """
        self .status ='completed'
        # ERSIN completed_at / recent endpoint için end_time'ı mutlaka doldur
        if self .end_time is None :
            self .end_time =datetime .utcnow ()

    def fail_analysis (self ,message :str ):
        """
        Analizi başarısız olarak işaretler.
        
        Args:
            message: Hata mesajı
        """
        self .status ='failed'
        self .error_message =message 
        # ERSIN failed analizlerde de bitiş zamanı yazılsın
        if self .end_time is None :
            self .end_time =datetime .utcnow ()

    def to_dict (self )->dict [str ,Any ]:
        """
        Analiz nesnesini sözlük formatında döndürür.
        
        Returns:
            dict: Analizin tüm özellikleriyle sözlük temsili
        """
        # ERSIN File bilgilerini güvenli şekilde al
        file_info =None 
        if self .file :
            file_info ={
            'id':self .file .id ,
            'filename':self .file .filename ,
            'original_filename':self .file .original_filename ,
            'file_type':self .file .file_type ,
            'file_size':self .file .file_size ,
            'mime_type':self .file .mime_type 
            }

        return {
        'id':self .id ,
        'file_id':self .file_id ,
        'file_info':file_info ,
        'status':self .status ,
        'start_time':self .start_time .isoformat ()if self .start_time is not None else None ,
        'end_time':self .end_time .isoformat ()if self .end_time is not None else None ,
        'created_at':self .created_at .isoformat ()if getattr (self ,"created_at",None )else None ,
        'error_message':self .error_message ,
        'frames_per_second':self .frames_per_second ,
        'frames_analyzed':self .frames_analyzed ,
        'include_age_analysis':self .include_age_analysis ,
        'websocket_session_id':self .websocket_session_id ,
        'is_cancelled':self .is_cancelled ,
        'overall_violence_score':self .overall_violence_score ,
        'overall_adult_content_score':self .overall_adult_content_score ,
        'overall_harassment_score':self .overall_harassment_score ,
        'overall_weapon_score':self .overall_weapon_score ,
        'overall_drug_score':self .overall_drug_score ,
        'overall_safe_score':self .overall_safe_score ,
        'highest_risk_frame':self .highest_risk_frame ,
        'highest_risk_frame_timestamp':self .highest_risk_frame_timestamp ,
        'highest_risk_score':self .highest_risk_score ,
        'highest_risk_category':self .highest_risk_category ,
        'category_specific_highest_risks_data':self .category_specific_highest_risks_data 
        }


class ContentDetection (db .Model ):
    """
    Her bir kare veya resim için içerik analiz sonuçlarını saklar.
    Şiddet, yetişkin içerik, taciz, silah ve madde kullanımı kategorilerindeki
    tespit skorlarını ve tespit edilen nesneleri içerir.
    """
    __tablename__ ='content_detections'

    id =db .Column (db .Integer ,primary_key =True )
    analysis_id =db .Column (db .String (36 ),db .ForeignKey ('analyses.id'),nullable =False )
    frame_path =db .Column (db .String (255 ))# ERSIN Analiz edilen karenin dosya yolu
    frame_timestamp =db .Column (db .Float )# ERSIN Karenin video içindeki zaman damgası (saniye)
    frame_index =db .Column (db .Integer ,nullable =True )# ERSIN Kare indeksi (sıra numarası)

    # ERSIN Kategorik skorlar (0-1 arası)
    violence_score =db .Column (db .Float ,default =0 )
    adult_content_score =db .Column (db .Float ,default =0 )
    harassment_score =db .Column (db .Float ,default =0 )
    weapon_score =db .Column (db .Float ,default =0 )
    drug_score =db .Column (db .Float ,default =0 )
    safe_score =db .Column (db .Float ,default =0 )# ERSIN Güvenli içerik skoru

    # ERSIN Tespit edilen nesneler JSON formatında
    _detected_objects =db .Column (db .Text )

    def get_detected_objects (self ):
        """Tespit edilen nesneleri JSON formatında döndürür."""
        if self ._detected_objects :
            try :
                return json .loads (self ._detected_objects )
            except :
                return []
        return []

    def set_detected_objects (self ,objects ):
        """Tespit edilen nesneleri JSON formatında saklar."""
        if objects :
            self ._detected_objects =json .dumps (objects ,cls =NumPyJSONEncoder )
        else :
            self ._detected_objects =None 

    @property 
    def detected_objects_json (self )->str :
        """Get the detected objects JSON string."""
        return self ._detected_objects or ""

    @detected_objects_json .setter 
    def detected_objects_json (self ,value :str ):
        """Set the detected objects JSON string."""
        self ._detected_objects =value 

    def to_dict (self )->dict [str ,Any ]:
        """
        İçerik tespitini sözlük formatında döndürür.
        
        Returns:
            dict: İçerik tespitinin tüm özellikleriyle sözlük temsili
        """
        return {
        'id':self .id ,
        'analysis_id':self .analysis_id ,
        'frame_path':self .frame_path ,
        'frame_timestamp':self .frame_timestamp ,
        'frame_index':self .frame_index ,
        'violence_score':self .violence_score ,
        'adult_content_score':self .adult_content_score ,
        'harassment_score':self .harassment_score ,
        'weapon_score':self .weapon_score ,
        'drug_score':self .drug_score ,
        'safe_score':self .safe_score ,
        'detected_objects':self .get_detected_objects ()
        }


class AgeEstimation (db .Model ):
    """
    Yaş tahmin sonuçlarını saklar.
    - Her bir kişi için yaş tahmini, güven skoru ve kişi izleme bilgileri içerir.
    """
    __tablename__ ='age_estimations'

    id =db .Column (db .Integer ,primary_key =True )
    analysis_id =db .Column (db .String (36 ),db .ForeignKey ('analyses.id'),nullable =False )
    person_id =db .Column (db .String (36 ),nullable =False ,index =True )# ERSIN Kişi takip ID'si
    frame_path =db .Column (db .String (255 ))# ERSIN Yaş tahmini yapılan karenin dosya yolu
    frame_timestamp =db .Column (db .Float )# ERSIN Karenin video içindeki zaman damgası (saniye)
    frame_index =db .Column (db .Integer ,nullable =True )# ERSIN Kare indeksi (sıra numarası)

    # ERSIN Yaş tahmini sonuçları
    estimated_age =db .Column (db .Float ,nullable =False )
    confidence_score =db .Column (db .Float ,nullable =False )

    # ERSIN Yüz tespit bilgileri
    face_bbox =db .Column (db .String (255 ))# ERSIN Yüz bounding box koordinatları (JSON formatında)
    face_landmarks =db .Column (db .Text )# ERSIN Yüz işaretleri (JSON formatında)

    # ERSIN Yüz embedding'i (kişi tanıma için)
    embedding =db .Column (db .Text )# ERSIN Virgül ile ayrılmış float string

    # ERSIN Yaş aralığı bilgisi (opsiyonel)
    age_range =db .Column (db .String (20 ))# ERSIN Örn: "18-25", "26-35", vb.

    # ERSIN İşlenmiş görsel yolu (overlay'li resim)
    processed_image_path =db .Column (db .String (255 ))# ERSIN Overlay'li resmin yolu

    def to_dict (self )->dict [str ,Any ]:
        """
        Yaş tahminini sözlük formatında döndürür.
        
        Returns:
            dict: Yaş tahmininin tüm özellikleriyle sözlük temsili
        """
        return {
        'id':self .id ,
        'analysis_id':self .analysis_id ,
        'person_id':self .person_id ,
        'frame_path':self .frame_path ,
        'frame_timestamp':self .frame_timestamp ,
        'frame_index':self .frame_index ,
        'estimated_age':self .estimated_age ,
        'confidence_score':self .confidence_score ,
        'face_bbox':self .face_bbox ,
        'face_landmarks':self .face_landmarks ,
        'embedding':self .embedding ,
        'processed_image_path':self .processed_image_path 
        }

    def get_face_bbox (self )->dict [str ,Any ]:
        """Yüz bounding box koordinatlarını sözlük formatında döndürür."""
        if self .face_bbox :
            try :
                return json .loads (self .face_bbox )
            except :
                return {}
        return {}

    def set_face_bbox (self ,bbox :dict [str ,Any ]):
        """Yüz bounding box koordinatlarını JSON formatında saklar."""
        if bbox :
            self .face_bbox =json .dumps (bbox )
        else :
            self .face_bbox =None 

    def get_face_landmarks (self )->dict [str ,Any ]:
        """Yüz işaretlerini sözlük formatında döndürür."""
        if self .face_landmarks :
            try :
                return json .loads (self .face_landmarks )
            except :
                return {}
        return {}

    def set_face_landmarks (self ,landmarks :dict [str ,Any ]):
        """Yüz işaretlerini JSON formatında saklar."""
        if landmarks :
            self .face_landmarks =json .dumps (landmarks ,cls =NumPyJSONEncoder )
        else :
            self .face_landmarks =None 