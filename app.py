from wsgi import app, socketio
import os

if __name__ == '__main__':
    # Ana klasörler için statik servis yapalım
    app.config['STORAGE_PATH'] = os.path.join(os.path.dirname(__file__), 'storage')
    
    # Debug modunda çalıştır
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 