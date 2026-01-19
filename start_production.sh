#!/bin/bash
# WSANALIZ Production Start Script

echo "Starting WSANALIZ Production Server..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export FLASK_ENV=production
export FLASK_DEBUG=0

# Start the application with eventlet worker for SocketIO support
# SocketIO requires eventlet or gevent worker, not standard gunicorn workers
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --worker-class eventlet \
         --worker-connections 1000 \
         --timeout 300 \
         --keep-alive 5 \
         --log-level info \
         --access-logfile - \
         --error-logfile - \
         wsgi:app

echo "WSANALIZ Server started on port 5000"
