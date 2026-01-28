from datetime import datetime 
from app import db 
from sqlalchemy import Column ,Integer ,String ,DateTime ,Boolean ,JSON 

class ModelVersion (db .Model ):
    """
    Eğitilmiş modellerin sürüm bilgilerini saklar.
    Bu tablo, model eğitim geçmişini ve metriklerini takip etmeyi sağlar.
    """
    __tablename__ ='model_versions'

    id =db .Column (db .Integer ,primary_key =True )
    model_type =db .Column (db .String (50 ),nullable =False )# ERSIN 'content' veya 'age'
    version =db .Column (db .Integer ,nullable =False )
    version_name =db .Column (db .String (100 ),nullable =True )# ERSIN Örn: "v1_20240101_120000"
    created_at =db .Column (db .DateTime ,default =datetime .now )
    metrics =db .Column (db .JSON )# ERSIN Doğruluk, F1, Kesinlik, Geri çağırma skorları
    is_active =db .Column (db .Boolean ,default =True )
    training_samples =db .Column (db .Integer )# ERSIN Eğitim için kullanılan örnek sayısı
    validation_samples =db .Column (db .Integer )# ERSIN Doğrulama için kullanılan örnek sayısı
    epochs =db .Column (db .Integer )# ERSIN Eğitimde kullanılan epoch sayısı

    # ERSIN Model dosya yolları
    file_path =db .Column (db .String (255 ))
    weights_path =db .Column (db .String (255 ))

    # ERSIN İlişkili geri bildirimler (eğitimde kullanılan)
    used_feedback_ids =db .Column (db .JSON )# ERSIN Eğitimde kullanılan geri bildirim ID'leri

    def __repr__ (self ):
        return f"<ModelVersion(id={self .id }, model_type='{self .model_type }', version={self .version }, active={self .is_active })>"