from app import create_app, socketio
import logging
import os

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO ve WARNING loglarını gizle
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Sadece ERROR loglarını göster

app = create_app()

if __name__ == '__main__':
    # Werkzeug HTTP request loglarını kapat
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Development sunucusu için
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, log_output=False) 