from app import create_app
import logging
import os

"""
WSGI sunucusu için giriş noktası.
- Flask uygulamasını WSGI uyumlu sunucularla başlatmak için kullanılır.
"""

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO ve WARNING loglarını gizle
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Sadece ERROR loglarını göster

app = create_app()

if __name__ == '__main__':
    # Werkzeug HTTP request loglarını kapat
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Development sunucusu için - SocketIO kaldırıldı
    app.run(debug=True, host='0.0.0.0', port=5000) 