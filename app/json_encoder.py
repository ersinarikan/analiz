import json 
import numpy as np 

class NumPyJSONEncoder (json .JSONEncoder ):
    """NumPy türlerini standart Python türlerine dönüştüren özel JSON kodlayıcı."""

    def default (self ,o ):# ERSIN Base class uses 'o' parameter name
        obj =o # ERSIN Alias for clarity
        # ERSIN Type checking: use type() instead of isinstance for generic types
        if isinstance (obj ,np .integer )or type (obj ).__name__ in ('int8','int16','int32','int64'):
            return int (obj )
        elif isinstance (obj ,np .floating )or type (obj ).__name__ in ('float16','float32','float64'):
            return float (obj )
        elif isinstance (obj ,np .ndarray ):
            return obj .tolist ()
        elif isinstance (obj ,np .bool_ ):
            return bool (obj )
        elif isinstance (obj ,complex )or type (obj ).__name__ in ('complex128','complex64'):
            return {'real':obj .real ,'imag':obj .imag }
        elif hasattr (obj ,'dtype')and hasattr (obj ,'item'):
        # ERSIN Genel NumPy skalar tipi için item() metodunu kullan
            return obj .item ()
        elif hasattr (obj ,'to_dict'):
            return obj .to_dict ()
        return super (NumPyJSONEncoder ,self ).default (obj )

        # ERSIN Alias için Flask compatibility
CustomJSONEncoder =NumPyJSONEncoder 

def json_dumps_numpy (obj ):
    """NumPy dizilerini içeren nesneleri JSON'a dönüştürür."""
    return json .dumps (obj ,cls =NumPyJSONEncoder )

def json_dump_numpy (obj ,fp ):
    """NumPy dizilerini içeren nesneleri bir dosyaya JSON olarak yazar."""
    return json .dump (obj ,fp ,cls =NumPyJSONEncoder )