from flask import current_app
from app import db
import logging

logger = logging.getLogger(__name__)

def save_to_db(model_instance):
    """
    Veritabanı modelini kaydeder.
    
    Args:
        model_instance: Kaydedilecek model nesnesi
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
        
    Raises:
        Exception: Veritabanı işlemi sırasında oluşan hatalar
    """
    try:
        db.session.add(model_instance)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Veritabanına kaydetme hatası: {str(e)}")
        raise

def query_db(model, **kwargs):
    """
    Veritabanında belirtilen modeli filtreleyerek arar.
    
    Args:
        model: Sorgulanacak model sınıfı
        **kwargs: Sorgu filtreleri
        
    Returns:
        model ya da None: Eşleşen model örneği veya bulunamazsa None
    """
    try:
        return model.query.filter_by(**kwargs).first()
    except Exception as e:
        current_app.logger.error(f"Veritabanı sorgu hatası: {str(e)}")
        return None
        
def query_all_db(model, **kwargs):
    """
    Veritabanında belirtilen modeli filtreleyerek arar ve tüm sonuçları döndürür.
    
    Args:
        model: Sorgulanacak model sınıfı
        **kwargs: Sorgu filtreleri
        
    Returns:
        list: Eşleşen model örneklerinin listesi
    """
    try:
        return model.query.filter_by(**kwargs).all()
    except Exception as e:
        current_app.logger.error(f"Veritabanı sorgu hatası: {str(e)}")
        return []

def update_db(model_instance, **kwargs):
    """
    Veritabanı model örneğini günceller.
    
    Args:
        model_instance: Güncellenecek model örneği
        **kwargs: Güncellenecek alanlar ve değerleri
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
    """
    try:
        for key, value in kwargs.items():
            setattr(model_instance, key, value)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Veritabanı güncelleme hatası: {str(e)}")
        return False
        
def delete_from_db(model_instance):
    """
    Veritabanından model örneğini siler.
    
    Args:
        model_instance: Silinecek model örneği
        
    Returns:
        bool: İşlem başarılı ise True, aksi halde False
    """
    try:
        db.session.delete(model_instance)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Veritabanından silme hatası: {str(e)}")
        return False 