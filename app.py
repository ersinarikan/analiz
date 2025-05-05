from app import create_app, socketio, initialize_app

if __name__ == "__main__":
    app = create_app()
    initialize_app(app)  # Sadece ana süreçte çalıştırılacak
    socketio.run(app, debug=True, host="0.0.0.0", port=5000) 