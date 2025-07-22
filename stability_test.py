#!/usr/bin/env python3
"""
WebSocket Stabilite Testi - Uzun süreli bağlantı testi
"""

import socketio
import time
import threading

# SocketIO client oluştur
sio = socketio.Client()

test_results = {
    'connected_count': 0,
    'pong_count': 0,
    'disconnected_count': 0,
    'errors': []
}

@sio.event
def connect():
    test_results['connected_count'] += 1
    print(f"✅ BAĞLANTI #{test_results['connected_count']}")

@sio.event
def disconnect():
    test_results['disconnected_count'] += 1
    print(f"❌ BAĞLANTI KESİLDİ #{test_results['disconnected_count']}")

@sio.event
def pong(data):
    test_results['pong_count'] += 1
    print(f"🏓 PONG #{test_results['pong_count']}")

def ping_sender():
    """Background thread'de ping gönder"""
    try:
        while sio.connected:
            sio.emit('ping', f'stability_test_{time.time()}')
            time.sleep(10)  # 10 saniyede bir ping
    except Exception as e:
        test_results['errors'].append(f"Ping sender error: {e}")
        print(f"❌ Ping sender hatası: {e}")

def main():
    print("🧪 WebSocket Stabilite Testi Başlatılıyor...")
    print("⏰ 2 dakika boyunca bağlantıyı test edeceğiz...")
    
    try:
        # Bağlan
        sio.connect('http://localhost:5000')
        time.sleep(2)
        
        # Background ping thread başlat
        ping_thread = threading.Thread(target=ping_sender, daemon=True)
        ping_thread.start()
        
        # 2 dakika boyunca bağlantıyı koru
        test_duration = 120  # 2 dakika
        print(f"🔄 {test_duration} saniye stabilite testi başlıyor...")
        
        for i in range(test_duration):
            if not sio.connected:
                print("❌ Bağlantı kesildi, yeniden bağlanmaya çalışıyorum...")
                try:
                    sio.connect('http://localhost:5000')
                except Exception as e:
                    test_results['errors'].append(f"Reconnect error: {e}")
            
            time.sleep(1)
            
            # Her 30 saniyede bir rapor
            if (i + 1) % 30 == 0:
                print(f"📊 {i+1}s - Ping: {test_results['pong_count']}, Disconnect: {test_results['disconnected_count']}")
        
        print("\n📋 FİNAL RAPOR:")
        print(f"🔗 Toplam bağlantı: {test_results['connected_count']}")
        print(f"🏓 Başarılı PONG: {test_results['pong_count']}")
        print(f"❌ Disconnect sayısı: {test_results['disconnected_count']}")
        print(f"🚨 Hata sayısı: {len(test_results['errors'])}")
        
        if test_results['errors']:
            print("🔍 HATALAR:")
            for error in test_results['errors']:
                print(f"   - {error}")
        
        # Stability score hesapla
        expected_pings = test_duration // 10  # 10 saniyede bir ping
        stability_score = (test_results['pong_count'] / expected_pings) * 100 if expected_pings > 0 else 0
        
        print(f"\n🎯 STABİLİTE SKORU: {stability_score:.1f}%")
        
        if stability_score >= 90:
            print("🎉 MÜKEMMEL STABİLİTE!")
        elif stability_score >= 70:
            print("✅ İYİ STABİLİTE")
        else:
            print("⚠️ STABİLİTE SORUNU VAR")
        
        # Bağlantıyı kapat
        sio.disconnect()
        
    except Exception as e:
        print(f"❌ HATA: {e}")
        test_results['errors'].append(f"Main error: {e}")

if __name__ == "__main__":
    main() 