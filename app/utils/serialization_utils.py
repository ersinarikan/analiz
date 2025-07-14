import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def convert_numpy_types_to_python(obj, debug_path=""):
    """
    Tüm NumPy türlerini Python standart türlerine dönüştürür.
    
    Args:
        obj: Dönüştürülecek nesne
        debug_path: Hata ayıklama için nesne yolu
        
    Returns:
        obj: Python standart tiplerinden oluşan nesne
    """
    try:
        # NumPy array'i Python list'e çevir
        if isinstance(obj, np.ndarray):
            return convert_numpy_types_to_python(obj.tolist(), f"{debug_path}[ndarray]")
            
        # NumPy sayısal tiplerini Python tiplerine çevir
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Dict içeriğini recursive olarak dönüştür
        elif isinstance(obj, dict):
            return {str(convert_numpy_types_to_python(k, f"{debug_path}[key:{k}]")): 
                    convert_numpy_types_to_python(v, f"{debug_path}[{k}]") 
                   for k, v in obj.items()}
            
        # List/tuple içeriğini recursive olarak dönüştür
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types_to_python(item, f"{debug_path}[{i}]") 
                   for i, item in enumerate(obj)]
            
        # Diğer özel NumPy tipleri için
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            logger.debug(f"Converting NumPy scalar at {debug_path}: {type(obj)}")
            return obj.item()
            
        # Serileştirilemeyecek objeleri string'e çevir
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            logger.warning(f"Converting non-serializable object at {debug_path}: {type(obj)} to string")
            return str(obj)
            
        # Zaten Python tipi ise olduğu gibi bırak
        return obj
        
    except Exception as e:
        logger.error(f"Error converting NumPy type at {debug_path}: {str(e)}")
        # Hata durumunda string olarak dön
        return str(obj)

def debug_serialization(obj, prefix=""):
    """
    Serileştirilemeyen objeleri tespit etmek için objeyi derinlemesine inceler.
    
    Args:
        obj: İncelenecek nesne
        prefix: Hata ayıklama için prefix
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return  # Temel tipler sorun oluşturmaz
    
    # NumPy tipleri kontrolü
    if isinstance(obj, np.integer):
        logger.error(f"{prefix}: NumPy integer bulundu: {obj}, type: {type(obj)}")
    elif isinstance(obj, np.floating):
        logger.error(f"{prefix}: NumPy float bulundu: {obj}, type: {type(obj)}")
    elif isinstance(obj, np.ndarray):
        logger.error(f"{prefix}: NumPy array bulundu, shape: {obj.shape}, dtype: {obj.dtype}")
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        logger.error(f"{prefix}: NumPy scalar bulundu: {obj}, type: {type(obj)}")
    
    # İç içe yapıları incele
    if isinstance(obj, dict):
        for k, v in obj.items():
            debug_serialization(k, f"{prefix}[key]")
            debug_serialization(v, f"{prefix}[{k}]")
    
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            debug_serialization(item, f"{prefix}[{i}]")

def serialize_to_json(obj):
    """
    NumPy tiplerini Python tiplerine çevirip JSON'a serileştirir.
    
    Args:
        obj: Serileştirilecek nesne
        
    Returns:
        str: JSON formatında string
    """
    try:
        converted_obj = convert_numpy_types_to_python(obj)
        return json.dumps(converted_obj)
    except Exception as e:
        logger.error(f"Serialization error: {str(e)}")
        # Hata durumunda serileştirme hatasını debug et
        debug_serialization(obj)
        # Boş nesne döndür
        return "{}" 