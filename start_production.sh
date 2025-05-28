#!/bin/bash
# CLIP Training Production Start Script

echo "Starting CLIP Training Production Server..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export FLASK_ENV=production
export FLASK_DEBUG=0

# Start the application
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 wsgi:app

echo "CLIP Training Server started on port 5000"
