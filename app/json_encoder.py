import json
import numpy as np

class NumPyJSONEncoder(json.JSONEncoder):
    """NumPy türlerini standart Python türlerine dönüştüren özel JSON kodlayıcı."""
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (complex, np.complex128, np.complex64)):
            return {'real': obj.real, 'imag': obj.imag}
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            # Genel NumPy skalar tipi için item() metodunu kullan
            return obj.item()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super(NumPyJSONEncoder, self).default(obj)

# Alias for Flask compatibility
CustomJSONEncoder = NumPyJSONEncoder

def json_dumps_numpy(obj):
    """NumPy dizilerini içeren nesneleri JSON'a dönüştürür."""
    return json.dumps(obj, cls=NumPyJSONEncoder)

def json_dump_numpy(obj, fp):
    """NumPy dizilerini içeren nesneleri bir dosyaya JSON olarak yazar."""
    return json.dump(obj, fp, cls=NumPyJSONEncoder) 