#!/usr/bin/env python3
"""
WebSocket Test Script - Otomatik WebSocket Bağlantı ve Event Test
Bu script WebSocket bağlantısını test eder ve sorunları otomatik bulur.
"""

import socketio
import time
import requests
import json
import threading
import sys
import os

class WebSocketTester:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.sio = socketio.Client()
        self.events_received = {}
        self.errors = []
        self.connected = False
        
        # Event listeners kurulumu
        self.setup_event_listeners()
        
    def setup_event_listeners(self):
        """Event listener'ları kurar"""
        
        @self.sio.event
        def connect():
            print("✅ WebSocket bağlantısı başarılı!")
            self.connected = True
            
        @self.sio.event
        def disconnect():
            print("❌ WebSocket bağlantısı kesildi!")
            self.connected = False
            
        @self.sio.event
        def connect_error(data):
            print(f"❌ Bağlantı hatası: {data}")
            self.errors.append(f"connect_error: {data}")
        
        # Catch-all event listener
        @self.sio.on('*')
        def catch_all(event, *args):
            print(f"🎧 GENEL EVENT ALINDI: {event} - Args: {args}")
            
        @self.sio.event
        def connected(data):
            print(f"🎉 CONNECTED event alındı: {data}")
            self.events_received['connected'] = data
            
        @self.sio.event
        def pong(data):
            print(f"🏓 PONG event alındı: {data}")
            self.events_received['pong'] = data
            
        @self.sio.event
        def joined_analysis(data):
            print(f"🎯 JOINED_ANALYSIS event alındı: {data}")
            self.events_received['joined_analysis'] = data
            
        @self.sio.event
        def analysis_progress(data):
            print(f"📊 ANALYSIS_PROGRESS event alındı: {data}")
            if 'analysis_progress' not in self.events_received:
                self.events_received['analysis_progress'] = []
            self.events_received['analysis_progress'].append(data)
    
    def connect_to_server(self):
        """Sunucuya bağlanır"""
        try:
            print(f"🔄 {self.server_url} adresine bağlanıyor...")
            self.sio.connect(self.server_url)
            time.sleep(2)  # Bağlantının stabilleşmesi için bekle
            return True
        except Exception as e:
            print(f"❌ Bağlantı hatası: {e}")
            self.errors.append(f"connection_error: {e}")
            return False
    
    def test_ping(self):
        """Ping event'ini test eder"""
        print("\n🏓 PING testi başlatılıyor...")
        try:
            print("🔄 Ping event gönderiliyor...")
            self.sio.emit('ping', 'test')
            print("⏳ PONG event'ini bekliyorum... (5 saniye)")
            time.sleep(5)  # Pong'un gelmesi için bekle
            
            print(f"📨 Alınan tüm event'ler: {list(self.events_received.keys())}")
            
            if 'pong' in self.events_received:
                print("✅ PING-PONG testi başarılı!")
                return True
            else:
                print("❌ PONG event alınamadı!")
                print("🔍 Debug: Tüm alınan event'ler:")
                for event_name, event_data in self.events_received.items():
                    print(f"   📧 {event_name}: {event_data}")
                self.errors.append("ping_test_failed: No pong received")
                return False
        except Exception as e:
            print(f"❌ PING testi hatası: {e}")
            self.errors.append(f"ping_test_error: {e}")
            return False
    
    def upload_test_file(self):
        """Test dosyası yükler"""
        print("\n📁 Test dosyası yükleniyor...")
        try:
            # Basit bir test dosyası oluştur
            test_file_content = b"fake_image_content_for_test"
            
            files = {
                'files': ('test.jpg', test_file_content, 'image/jpeg')
            }
            
            response = requests.post(
                f"{self.server_url}/upload",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Dosya yüklendi: {result}")
                return result.get('files', [])
            else:
                print(f"❌ Dosya yükleme hatası: {response.status_code}")
                self.errors.append(f"file_upload_error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Dosya yükleme exception: {e}")
            self.errors.append(f"file_upload_exception: {e}")
            return []
    
    def start_analysis(self, file_id):
        """Analiz başlatır"""
        print(f"\n🔬 Analiz başlatılıyor: file_id={file_id}")
        try:
            data = {
                'file_id': file_id,
                'frames_per_second': 1,
                'include_age_analysis': True
            }
            
            response = requests.post(
                f"{self.server_url}/start_analysis",
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_id = result.get('analysis', {}).get('id')
                print(f"✅ Analiz başlatıldı: {analysis_id}")
                return analysis_id
            else:
                print(f"❌ Analiz başlatma hatası: {response.status_code}")
                self.errors.append(f"analysis_start_error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Analiz başlatma exception: {e}")
            self.errors.append(f"analysis_start_exception: {e}")
            return None
    
    def test_join_analysis(self, analysis_id):
        """Join analysis event'ini test eder"""
        print(f"\n🎯 JOIN_ANALYSIS testi başlatılıyor: {analysis_id}")
        try:
            self.sio.emit('join_analysis', {'analysis_id': analysis_id})
            time.sleep(5)  # Events'lerin gelmesi için bekle
            
            if 'joined_analysis' in self.events_received:
                print("✅ JOIN_ANALYSIS testi başarılı!")
                return True
            else:
                print("❌ JOINED_ANALYSIS event alınamadı!")
                self.errors.append("join_analysis_test_failed: No joined_analysis received")
                return False
        except Exception as e:
            print(f"❌ JOIN_ANALYSIS testi hatası: {e}")
            self.errors.append(f"join_analysis_test_error: {e}")
            return False
    
    def test_analysis_progress(self):
        """Analysis progress event'larını test eder"""
        print("\n📊 ANALYSIS_PROGRESS testi...")
        time.sleep(10)  # Analiz progress'inin gelmesi için bekle
        
        if 'analysis_progress' in self.events_received:
            progress_events = self.events_received['analysis_progress']
            print(f"✅ {len(progress_events)} adet ANALYSIS_PROGRESS event alındı!")
            for i, event in enumerate(progress_events):
                print(f"   📈 Event {i+1}: {event.get('progress')}% - {event.get('message')}")
            return True
        else:
            print("❌ ANALYSIS_PROGRESS event alınamadı!")
            self.errors.append("analysis_progress_test_failed: No analysis_progress received")
            return False
    
    def run_full_test(self):
        """Tam test sürecini çalıştırır"""
        print("🚀 WebSocket Full Test Başlatılıyor...")
        print("=" * 60)
        
        # 1. Bağlantı testi
        if not self.connect_to_server():
            return self.generate_report()
        
        # 2. Ping testi
        ping_success = self.test_ping()
        
        # 3. Dosya yükleme testi
        uploaded_files = self.upload_test_file()
        if not uploaded_files:
            return self.generate_report()
        
        file_id = uploaded_files[0].get('id')
        
        # 4. Analiz başlatma testi
        analysis_id = self.start_analysis(file_id)
        if not analysis_id:
            return self.generate_report()
        
        # 5. Join analysis testi
        join_success = self.test_join_analysis(analysis_id)
        
        # 6. Analysis progress testi
        progress_success = self.test_analysis_progress()
        
        # 7. Bağlantıyı kapat
        self.sio.disconnect()
        
        return self.generate_report()
    
    def generate_report(self):
        """Test raporu oluşturur"""
        print("\n" + "=" * 60)
        print("📋 TEST RAPORU")
        print("=" * 60)
        
        print(f"🔗 Bağlantı Durumu: {'✅ Başarılı' if self.connected else '❌ Başarısız'}")
        print(f"📨 Alınan Event'ler: {len(self.events_received)}")
        for event_name, event_data in self.events_received.items():
            if isinstance(event_data, list):
                print(f"   📧 {event_name}: {len(event_data)} adet")
            else:
                print(f"   📧 {event_name}: ✅")
        
        print(f"❌ Hatalar: {len(self.errors)}")
        for error in self.errors:
            print(f"   🚨 {error}")
        
        # Sonuç özeti
        success_score = len(self.events_received) / 4  # 4 ana event bekliyoruz
        if success_score >= 1.0 and len(self.errors) == 0:
            print("\n🎉 TÜM TESTLER BAŞARILI! WebSocket tamamen çalışıyor!")
            return True
        elif success_score >= 0.5:
            print(f"\n⚠️ KISMÎ BAŞARI! {success_score*100:.0f}% test geçti, ancak bazı sorunlar var.")
            return False
        else:
            print(f"\n💥 TESTLER BAŞARISIZ! {success_score*100:.0f}% test geçti.")
            return False

def check_server_running():
    """Sunucunun çalışıp çalışmadığını kontrol eder"""
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🧪 WebSocket Otomatik Test Başlatılıyor...")
    
    # Sunucu kontrolü
    if not check_server_running():
        print("❌ Flask sunucusu çalışmıyor! Lütfen önce 'python app.py' ile başlatın.")
        sys.exit(1)
    
    # Test başlat
    tester = WebSocketTester()
    success = tester.run_full_test()
    
    if success:
        print("\n🎯 SONUÇ: WebSocket tamamen çalışıyor!")
        sys.exit(0)
    else:
        print("\n🔧 SONUÇ: WebSocket'te sorunlar tespit edildi. Logları inceleyin.")
        sys.exit(1)

if __name__ == "__main__":
    main() 