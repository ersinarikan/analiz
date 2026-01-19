#!/bin/bash
# WSANALIZ Production Stop Script

echo "Stopping WSANALIZ Production Server..."

# Find and kill gunicorn processes
pkill -f "gunicorn.*wsgi:app"

# Also kill any remaining Python processes for this app
if [ -f wsanaliz.pid ]; then
    PID=$(cat wsanaliz.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID 2>/dev/null
        echo "Killed process $PID from PID file"
    fi
    rm -f wsanaliz.pid
fi

echo "WSANALIZ Server stopped"
