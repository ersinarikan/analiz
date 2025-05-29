#!/usr/bin/env python3

import requests
import json

def test_training_stats_api():
    """Training stats API endpoint'ini test eder"""
    
    try:
        # Age model stats
        print("🧪 Age model training stats test ediliyor...")
        response = requests.get('http://localhost:5000/api/model/training-stats/age')
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API yanıtı başarılı!")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            if 'stats' in data:
                stats = data['stats']
                print(f"\n📊 Özet:")
                print(f"  Manuel samples: {stats.get('manual_samples', 0)}")
                print(f"  Pseudo samples: {stats.get('pseudo_samples', 0)}")
                print(f"  Total samples: {stats.get('total_samples', 0)}")
                print(f"  Total feedbacks: {stats.get('total_feedbacks', 0)}")
        else:
            print(f"❌ API hatası: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Bağlantı hatası: Flask uygulaması çalışmıyor olabilir")
    except Exception as e:
        print(f"❌ Hata: {str(e)}")

if __name__ == "__main__":
    test_training_stats_api() 