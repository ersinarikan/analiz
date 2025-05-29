#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ JSON Encoder
====================

Bu modül NumPy array'leri ve diğer özel veri tiplerini JSON'a 
serileştirmek için özel encoder sınıfları içerir.
"""

import json
import numpy as np
from datetime import datetime

class NumPyJSONEncoder(json.JSONEncoder):
    """
    NumPy array'leri ve datetime nesnelerini JSON'a dönüştüren özel encoder.
    
    Bu encoder şu veri tiplerini destekler:
    - NumPy array'leri → Python listeleri
    - NumPy scalar değerleri → Python primitive tipleri  
    - datetime nesneleri → ISO format string
    """
    
    def default(self, obj):
        """
        Varsayılan JSON encoder'ının desteklemediği tipleri dönüştürür
        
        Args:
            obj: Serileştirilecek nesne
            
        Returns:
            JSON serileştirilebilir nesne
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        
        return super().default(obj)

def json_dumps_numpy(obj):
    """NumPy dizilerini içeren nesneleri JSON'a dönüştürür."""
    return json.dumps(obj, cls=NumPyJSONEncoder)

def json_dump_numpy(obj, fp):
    """NumPy dizilerini içeren nesneleri bir dosyaya JSON olarak yazar."""
    return json.dump(obj, fp, cls=NumPyJSONEncoder) 