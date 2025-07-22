#!/usr/bin/env python3
"""
Minimal Flask-SocketIO Test
"""

from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "Minimal SocketIO Test"

@socketio.on('connect')
def handle_connect():
    print("🎉 CLIENT BAĞLANDI!")
    emit('connected', {'message': 'Bağlantı başarılı!'})

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ CLIENT AYRIDI!")

@socketio.on('ping')
def handle_ping(data):
    print(f"🏓 PING ALINDI: {data}")
    emit('pong', {'message': 'PONG!', 'data': data})
    print("🔥 PONG GÖNDERİLDİ!")

if __name__ == '__main__':
    print("🚀 Minimal SocketIO Server Başlatılıyor...")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False) 