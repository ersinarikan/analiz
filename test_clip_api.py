#!/usr/bin/env python3
"""
CLIP API Test Scripti
"""

import requests
import json

def test_clip_api():
    try:
        # 1. Statistics test
        print("🧪 CLIP Training Statistics Test")
        print("=" * 40)
        
        r = requests.get('http://localhost:5000/api/clip-training/statistics')
        print(f"Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print("✅ API Başarılı!")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"❌ API Hatası: {r.text}")
            
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}")

if __name__ == "__main__":
    test_clip_api() 