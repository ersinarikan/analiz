# 📦 WSANALIZ Ubuntu DEB Paketi Kılavuzu

## 🏛️ T.C. Aile ve Sosyal Hizmetler Bakanlığı
### İçerik Güvenlik Analiz Sistemi - Ubuntu Paketi

---

## 🎯 Genel Bakış

Bu kılavuz, WSANALIZ uygulamasını Ubuntu sistemlerde `.deb` paketi olarak dağıtmak için hazırlanmıştır. Paket, uygulamayı sistem genelinde kurarak production ortamında çalışmaya hazır hale getirir.

## 📋 Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Ubuntu 20.04 LTS veya üzeri
- **RAM**: 4 GB (8 GB önerilen)
- **Disk**: 10 GB boş alan
- **CPU**: 2 çekirdek (4 çekirdek önerilen)
- **Python**: 3.8 veya üzeri

### Bağımlılıklar
```bash
# Otomatik olarak kurulacak paketler:
- python3
- python3-pip
- python3-venv
- nginx
- supervisor
```

## 🔨 DEB Paketi Oluşturma

### 1. Gerekli Araçları Kurun
```bash
sudo apt update
sudo apt install dpkg-dev build-essential
```

### 2. Paketi Oluşturun
```bash
# Script'i çalıştırın
python3 create_deb_package.py

# Oluşturulan paket
ls -la wsanaliz_1.0.0_all.deb
```

### 3. Paket İçeriğini Kontrol Edin
```bash
# Paket bilgilerini görüntüle
dpkg-deb --info wsanaliz_1.0.0_all.deb

# Paket içeriğini listele
dpkg-deb --contents wsanaliz_1.0.0_all.deb
```

## 📦 Kurulum

### Tek Komutla Kurulum
```bash
sudo dpkg -i wsanaliz_1.0.0_all.deb
sudo apt-get install -f  # Eksik bağımlılıkları çöz
```

### Manuel Bağımlılık Kurulumu (Gerekirse)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv nginx supervisor
```

## 🚀 Kurulum Sonrası

### Servis Durumunu Kontrol Edin
```bash
# Systemd servisi
sudo systemctl status wsanaliz

# Supervisor durumu
sudo supervisorctl status wsanaliz

# Nginx durumu
sudo systemctl status nginx
```

### Erişim Adresleri
- **Ana Uygulama**: http://localhost
- **Yönetim Paneli**: http://localhost/admin
- **API Dokümantasyonu**: http://localhost/api/docs

### Log Dosyaları
```bash
# Uygulama logları
sudo tail -f /var/log/wsanaliz/app.log

# Nginx logları
sudo tail -f /var/log/nginx/wsanaliz_access.log
sudo tail -f /var/log/nginx/wsanaliz_error.log

# Systemd logları
sudo journalctl -u wsanaliz -f
```

## ⚙️ Konfigürasyon

### Uygulama Ayarları
```bash
# Ana konfigürasyon dosyası
sudo nano /opt/wsanaliz/config.py

# Çevresel değişkenler
sudo nano /etc/systemd/system/wsanaliz.service
```

### Nginx Ayarları
```bash
# Site konfigürasyonu
sudo nano /etc/nginx/sites-available/wsanaliz

# Nginx'i yeniden başlat
sudo systemctl reload nginx
```

### Supervisor Ayarları
```bash
# Supervisor konfigürasyonu
sudo nano /etc/supervisor/conf.d/wsanaliz.conf

# Konfigürasyonu yeniden yükle
sudo supervisorctl reread
sudo supervisorctl update
```

## 🔧 Yönetim Komutları

### Servis Yönetimi
```bash
# Servisi başlat
sudo systemctl start wsanaliz

# Servisi durdur
sudo systemctl stop wsanaliz

# Servisi yeniden başlat
sudo systemctl restart wsanaliz

# Otomatik başlatmayı etkinleştir
sudo systemctl enable wsanaliz

# Otomatik başlatmayı devre dışı bırak
sudo systemctl disable wsanaliz
```

### Supervisor ile Yönetim
```bash
# Uygulamayı başlat
sudo supervisorctl start wsanaliz

# Uygulamayı durdur
sudo supervisorctl stop wsanaliz

# Uygulamayı yeniden başlat
sudo supervisorctl restart wsanaliz

# Durumu kontrol et
sudo supervisorctl status wsanaliz
```

### Veritabanı Yönetimi
```bash
# Veritabanını sıfırla
sudo -u wsanaliz /opt/wsanaliz/venv/bin/python /opt/wsanaliz/app.py --reset-db

# Backup oluştur
sudo -u wsanaliz cp /var/lib/wsanaliz/wsanaliz.db /var/lib/wsanaliz/backup_$(date +%Y%m%d).db
```

## 🔄 Güncelleme

### Yeni Sürüm Kurulumu
```bash
# Eski sürümü kaldır (veriler korunur)
sudo apt remove wsanaliz

# Yeni paketi kur
sudo dpkg -i wsanaliz_1.1.0_all.deb
sudo apt-get install -f
```

### Manuel Güncelleme
```bash
# Servisi durdur
sudo systemctl stop wsanaliz

# Kodu güncelle
sudo -u wsanaliz git -C /opt/wsanaliz pull

# Bağımlılıkları güncelle
sudo -u wsanaliz /opt/wsanaliz/venv/bin/pip install -r /opt/wsanaliz/requirements.txt

# Servisi başlat
sudo systemctl start wsanaliz
```

## 🗑️ Kaldırma

### Tam Kaldırma
```bash
# Paketi kaldır
sudo apt remove wsanaliz

# Konfigürasyon dosyalarını da kaldır
sudo apt purge wsanaliz

# Kullanıcı verilerini temizle (İsteğe bağlı)
sudo rm -rf /var/lib/wsanaliz
sudo rm -rf /var/log/wsanaliz
sudo userdel wsanaliz
sudo groupdel wsanaliz
```

## 🛡️ Güvenlik

### Firewall Ayarları
```bash
# HTTP trafiğine izin ver
sudo ufw allow 80/tcp

# HTTPS trafiğine izin ver (SSL kurulumunda)
sudo ufw allow 443/tcp

# SSH erişimini koru
sudo ufw allow 22/tcp

# Firewall'ı etkinleştir
sudo ufw enable
```

### SSL Sertifikası (Let's Encrypt)
```bash
# Certbot kur
sudo apt install certbot python3-certbot-nginx

# SSL sertifikası al
sudo certbot --nginx -d yourdomain.com

# Otomatik yenileme testi
sudo certbot renew --dry-run
```

## 📊 İzleme ve Performans

### Sistem Kaynaklarını İzleme
```bash
# CPU ve RAM kullanımı
htop

# Disk kullanımı
df -h

# Ağ trafiği
sudo netstat -tulpn | grep :80
```

### Uygulama Metrikleri
```bash
# Aktif bağlantılar
sudo ss -tulpn | grep :5000

# Log analizi
sudo tail -n 100 /var/log/wsanaliz/app.log | grep ERROR
```

## 🆘 Sorun Giderme

### Yaygın Sorunlar

#### 1. Servis Başlamıyor
```bash
# Detaylı log kontrol et
sudo journalctl -u wsanaliz -n 50

# Konfigürasyon dosyasını kontrol et
sudo -u wsanaliz /opt/wsanaliz/venv/bin/python -c "import config"
```

#### 2. Nginx 502 Hatası
```bash
# Uygulama çalışıyor mu?
sudo systemctl status wsanaliz

# Port dinleniyor mu?
sudo netstat -tulpn | grep :5000
```

#### 3. Veritabanı Hatası
```bash
# Veritabanı dosyası var mı?
ls -la /var/lib/wsanaliz/

# İzinler doğru mu?
sudo chown -R wsanaliz:wsanaliz /var/lib/wsanaliz/
```

#### 4. Yüksek Bellek Kullanımı
```bash
# Bellek kullanımını kontrol et
sudo ps aux | grep wsanaliz

# Supervisor ile yeniden başlat
sudo supervisorctl restart wsanaliz
```

## 📞 Destek

### Log Toplama
```bash
# Sistem bilgilerini topla
sudo /opt/wsanaliz/collect_logs.sh > wsanaliz_debug.txt
```

### İletişim
- **E-posta**: bilgi@aile.gov.tr
- **Telefon**: 0312 705 50 00
- **Adres**: Adres Eskişehir Yolu 9. Km, Söğütözü/ANKARA

---

## 📝 Notlar

- Bu paket production ortamı için optimize edilmiştir
- Tüm loglar `/var/log/wsanaliz/` dizininde saklanır
- Veritabanı `/var/lib/wsanaliz/` dizininde korunur
- Konfigürasyon dosyaları `/opt/wsanaliz/` dizinindedir

**Son Güncelleme**: 2025-01-04
**Sürüm**: 1.0.0 