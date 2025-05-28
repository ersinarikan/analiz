#!/bin/bash
# CLIP Training Health Check Script

echo "CLIP Training Health Check..."

# Check if server is running
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/clip-training/statistics)

if [ "$response" = "200" ]; then
    echo "CLIP Training Server is healthy"
    exit 0
else
    echo "CLIP Training Server is not responding (HTTP: $response)"
    exit 1
fi
