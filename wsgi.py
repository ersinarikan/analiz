"""
WSGI Configuration for production deployment
"""
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import create_app

# Initialize the Flask application
app, socketio = create_app()  # Tuple'ı unpack et

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000) 