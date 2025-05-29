#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSANALIZ WSGI Production Entry Point
===================================

Bu dosya production ortamında (Gunicorn, uWSGI vb.) kullanılmak üzere
WSGI uygulamasını başlatır.

Kullanım:
    gunicorn --bind 0.0.0.0:5000 wsgi:app
    uwsgi --module wsgi:app --http :5000
"""

import os
import logging

# Production ortamı için TensorFlow loglarını minimize et
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Sadece FATAL logları

try:
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
except ImportError:
    pass

from app import create_app, socketio

# Production configuration
app = create_app(config_name='production')

if __name__ == '__main__':
    # Production server için gerekli ayarlar
    logging.basicConfig(level=logging.WARNING)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000) 