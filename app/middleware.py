#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ Middleware
==================

Bu modül Flask uygulamasının middleware bileşenlerini içerir.
İstek/yanıt işleme, JSON serialization ve diğer ara katman işlemleri
burada tanımlanır.
"""

from flask import Flask, jsonify, current_app
import numpy as np
from functools import wraps
import json
import traceback
from app.json_encoder import NumPyJSONEncoder

def register_json_middleware(app: Flask):
    """
    Flask uygulamasına JSON middleware'ini kaydeder.
    
    Bu middleware NumPy türlerini otomatik olarak JSON'a dönüştürür
    ve API yanıtlarında tutarlı format sağlar.
    
    Args:
        app: Flask uygulaması
    """
    
    @app.before_request
    def before_request():
        """
        Her istek öncesi çalışan middleware.
        İstek loglaması ve güvenlik kontrolleri burada yapılabilir.
        """
        pass
    
    @app.after_request  
    def after_request(response):
        """
        Her yanıt sonrası çalışan middleware.
        CORS başlıkları ve yanıt loglaması burada yapılır.
        
        Args:
            response: Flask yanıt nesnesi
            
        Returns:
            İşlenmiş yanıt nesnesi
        """
        # CORS başlıklarını ekle
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        
        return response
    
    # Flask'ın JSON encoder'ını NumPy destekli encoder ile değiştir
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