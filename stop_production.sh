#!/bin/bash
# CLIP Training Production Stop Script

echo "Stopping CLIP Training Production Server..."

# Find and kill gunicorn processes
pkill -f "gunicorn.*wsgi:app"

echo "CLIP Training Server stopped"
