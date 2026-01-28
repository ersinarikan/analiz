import json 
import logging 
import numpy as np 

logger =logging .getLogger (__name__ )

def convert_numpy_types_to_python (obj ,debug_path =""):
    """
    Tüm NumPy türlerini Python standart türlerine dönüştürür.
    
    Args:
        obj: Dönüştürülecek nesne
        debug_path: Hata ayıklama için nesne yolu
        
    Returns:
        obj: Python standart tiplerinden oluşan nesne
    """
    try :
    # ERSIN NumPy array'i Python list'e çevir
        if isinstance (obj ,np .ndarray ):
            return convert_numpy_types_to_python (obj .tolist (),f"{debug_path }[ndarray]")

            # ERSIN NumPy sayısal tiplerini Python tiplerine çevir
            # ERSIN isinstance with tuple of types works correctly
            # ERSIN np.integer and np.floating are abstract base classes, use type() for concrete types
        if isinstance (obj ,np .integer ):
            return int (obj )
            # ERSIN Check concrete integer types using type() instead of isinstance
        obj_type =type (obj )
        if obj_type is np .int8 or obj_type is np .int16 or obj_type is np .int32 or obj_type is np .int64 :
            return int (obj )
        if isinstance (obj ,np .floating ):
            return float (obj )
            # ERSIN Check concrete float types using type() instead of isinstance
            # ERSIN obj_type is already defined above, reuse it
        if obj_type is np .float16 or obj_type is np .float32 or obj_type is np .float64 :
            return float (obj )
        elif isinstance (obj ,np .bool_ ):
            return bool (obj )

            # ERSIN Dict içeriğini recursive olarak dönüştür
        elif isinstance (obj ,dict ):
            return {str (convert_numpy_types_to_python (k ,f"{debug_path }[key:{k }]")):
            convert_numpy_types_to_python (v ,f"{debug_path }[{k }]")
            for k ,v in obj .items ()}

            # ERSIN List/tuple içeriğini recursive olarak dönüştür
        elif isinstance (obj ,(list ,tuple )):
            return [convert_numpy_types_to_python (item ,f"{debug_path }[{i }]")
            for i ,item in enumerate (obj )]

            # ERSIN Diğer özel NumPy tipleri için
        elif hasattr (obj ,'dtype')and hasattr (obj ,'item'):
            logger .debug (f"Converting NumPy scalar at {debug_path }: {type (obj )}")
            return obj .item ()

            # ERSIN Serileştirilemeyecek objeleri string'e çevir
        elif not isinstance (obj ,(str ,int ,float ,bool ,type (None ))):
            logger .warning (f"Converting non-serializable object at {debug_path }: {type (obj )} to string")
            return str (obj )

            # ERSIN Zaten Python tipi ise olduğu gibi bırak
        return obj 

    except Exception as e :
        logger .error (f"Error converting NumPy type at {debug_path }: {str (e )}")
        # ERSIN Hata durumunda string olarak dön
        return str (obj )

def debug_serialization (obj ,prefix =""):
    """
    Serileştirilemeyen objeleri tespit etmek için objeyi derinlemesine inceler.
    
    Args:
        obj: İncelenecek nesne
        prefix: Hata ayıklama için prefix
    """
    if isinstance (obj ,(str ,int ,float ,bool ,type (None ))):
        return # ERSIN Temel tipler sorun oluşturmaz

        # ERSIN NumPy tipleri kontrolü
    if isinstance (obj ,np .integer ):
        logger .error (f"{prefix }: NumPy integer bulundu: {obj }, type: {type (obj )}")
    elif isinstance (obj ,np .floating ):
        logger .error (f"{prefix }: NumPy float bulundu: {obj }, type: {type (obj )}")
    elif isinstance (obj ,np .ndarray ):
        logger .error (f"{prefix }: NumPy array bulundu, shape: {obj .shape }, dtype: {obj .dtype }")
    elif hasattr (obj ,'dtype')and hasattr (obj ,'item'):
        logger .error (f"{prefix }: NumPy scalar bulundu: {obj }, type: {type (obj )}")

        # ERSIN İç içe yapıları incele
    if isinstance (obj ,dict ):
        for k ,v in obj .items ():
            debug_serialization (k ,f"{prefix }[key]")
            debug_serialization (v ,f"{prefix }[{k }]")

    if isinstance (obj ,(list ,tuple )):
        for i ,item in enumerate (obj ):
            debug_serialization (item ,f"{prefix }[{i }]")

def serialize_to_json (obj ):
    """
    NumPy tiplerini Python tiplerine çevirip JSON'a serileştirir.
    
    Args:
        obj: Serileştirilecek nesne
        
    Returns:
        str: JSON formatında string
    """
    try :
        converted_obj =convert_numpy_types_to_python (obj )
        return json .dumps (converted_obj )
    except Exception as e :
        logger .error (f"Serialization error: {str (e )}")
        # ERSIN Hata durumunda serileştirme hatasını debug et
        debug_serialization (obj )
        # ERSIN Boş nesne döndür
        return "{}"