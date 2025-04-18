from flask import Flask, jsonify, current_app
import numpy as np
from functools import wraps
import json
import traceback
from app.json_encoder import NumPyJSONEncoder

def register_json_middleware(app: Flask):
    """
    Flask uygulamasına JSON serialization middleware'i kaydeder.
    NumPy dizilerini ve diğer özel tipleri otomatik olarak dönüştürür.
    
    Args:
        app: Flask uygulaması
    """
    # Flask'ın varsayılan JSON encoder'ını özel encoder ile değiştir
    app.json_encoder = NumPyJSONEncoder
    
    # jsonify fonksiyonunu override et
    original_jsonify = jsonify
    
    @wraps(jsonify)
    def numpy_jsonify(*args, **kwargs):
        """NumPy tiplerini destekleyen jsonify versiyonu"""
        if args and kwargs:
            raise TypeError('jsonify() behavior undefined when passed both args and kwargs')
        if len(args) == 1:
            data = args[0]
        else:
            data = args or kwargs
            
        return app.response_class(
            json.dumps(data, cls=NumPyJSONEncoder) + '\n',
            mimetype=app.config['JSONIFY_MIMETYPE']
        )
    
    # Global jsonify fonksiyonunu değiştir
    import flask
    flask.jsonify = numpy_jsonify
    
    # Log mesajı ekle
    app.logger.info("NumPy JSON middleware başarıyla kaydedildi")
    
    return app 