#!/usr/bin/env python3
"""
Basit WebSocket Test - Sadece ping-pong
"""

import socketio
import time

# SocketIO client oluştur
sio = socketio.Client()

received_events = []

@sio.event
def connect():
    print("✅ BAĞLANTI BAŞARILI!")

@sio.event
def disconnect():
    print("❌ BAĞLANTI KESİLDİ!")

@sio.event
def pong(data):
    print(f"🏓 PONG ALINDI: {data}")
    received_events.append('pong')

@sio.event
def connected(data):
    print(f"🎉 CONNECTED ALINDI: {data}")
    received_events.append('connected')

@sio.event
def joined_analysis(data):
    print(f"🔍 JOINED_ANALYSIS ALINDI: {data}")
    received_events.append('joined_analysis')

# Tüm event'leri yakala
@sio.on('*')
def catch_all(event, *args):
    print(f"🎧 GENEL EVENT: {event} - {args}")
    received_events.append(f"catch_all_{event}")

def main():
    print("🧪 Basit WebSocket Test...")
    
    try:
        # Bağlan
        print("🔄 Bağlanıyor...")
        sio.connect('http://localhost:5000')  # Ana uygulama
        time.sleep(2)
        
        # Ping gönder
        print("🏓 PING gönderiliyor...")
        sio.emit('ping', 'test_data')
        time.sleep(3)
        
        # Join analysis test
        print("🔍 JOIN_ANALYSIS testi...")
        test_analysis_id = "test-12345-abcde"
        sio.emit('join_analysis', {'analysis_id': test_analysis_id})
        time.sleep(3)
        
        # Bekle
        print("⏳ 5 saniye daha bekliyorum...")
        time.sleep(5)
        
        # Sonuçları yazdır
        print(f"\n📋 SONUÇ:")
        print(f"📨 Alınan event sayısı: {len(received_events)}")
        print(f"📧 Event'ler: {received_events}")
        
        # Test sonuçları
        tests_passed = 0
        total_tests = 3
        
        if 'pong' in received_events:
            print("✅ PING-PONG BAŞARILI!")
            tests_passed += 1
        else:
            print("❌ PONG ALINAMADI!")
            
        if 'connected' in received_events:
            print("✅ CONNECTED BAŞARILI!")
            tests_passed += 1
        else:
            print("❌ CONNECTED ALINAMADI!")
            
        if 'joined_analysis' in received_events:
            print("✅ JOIN_ANALYSIS BAŞARILI!")
            tests_passed += 1
        else:
            print("❌ JOINED_ANALYSIS ALINAMADI!")
            
        print(f"\n🎯 SONUÇ: {tests_passed}/{total_tests} test başarılı ({tests_passed*100//total_tests}%)")
            
        # Bağlantıyı kapat
        sio.disconnect()
        
    except Exception as e:
        print(f"❌ HATA: {e}")

if __name__ == "__main__":
    main() 