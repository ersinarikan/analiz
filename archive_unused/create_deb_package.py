#!/usr/bin/env python3
"""
T.C. Aile ve Sosyal Hizmetler Bakanlığı
İçerik Güvenlik Analiz Sistemi - DEB Paketi Oluşturucu

Bu script, WSANALIZ uygulamasını Ubuntu için .deb paketi haline getirir.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile

class DebPackageBuilder:
    def __init__(self):
        self.app_name = "wsanaliz"
        self.version = "1.0.0"
        self.description = "T.C. Aile ve Sosyal Hizmetler Bakanlığı İçerik Güvenlik Analiz Sistemi"
        self.maintainer = "T.C. Aile ve Sosyal Hizmetler Bakanlığı <bilgi@aile.gov.tr>"
        self.architecture = "all"
        self.depends = "python3, python3-pip, python3-venv, nginx, supervisor"
        
        self.build_dir = Path("deb_build")
        self.package_dir = self.build_dir / f"{self.app_name}_{self.version}"
        
    def create_directory_structure(self):
        """DEB paketi için dizin yapısını oluştur"""
        print("📁 Dizin yapısı oluşturuluyor...")
        
        # Ana dizinleri oluştur
        dirs = [
            "DEBIAN",
            "opt/wsanaliz",
            "etc/nginx/sites-available",
            "etc/supervisor/conf.d",
            "etc/systemd/system",
            "usr/bin",
            "var/log/wsanaliz",
            "var/lib/wsanaliz"
        ]
        
        for dir_path in dirs:
            (self.package_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
    def create_control_file(self):
        """DEBIAN/control dosyasını oluştur"""
        print("📋 Control dosyası oluşturuluyor...")
        
        control_content = f"""Package: {self.app_name}
Version: {self.version}
Section: web
Priority: optional
Architecture: {self.architecture}
Depends: {self.depends}
Maintainer: {self.maintainer}
Description: {self.description}
 Dijital içeriklerin güvenlik analizi ve değerlendirmesi için
 geliştirilmiş resmi platform. Çocukların korunması ve aile güvenliği
 kapsamında içerik denetimi yapılmaktadır.
 .
 Bu paket aşağıdaki özellikleri içerir:
 - AI tabanlı içerik analizi
 - Çoklu format desteği (resim, video)
 - Web tabanlı yönetim paneli
 - Güvenli API erişimi
 - Detaylı raporlama sistemi
"""
        
        with open(self.package_dir / "DEBIAN" / "control", "w", encoding="utf-8") as f:
            f.write(control_content)
            
    def create_postinst_script(self):
        """Kurulum sonrası script oluştur"""
        print("🔧 Post-install scripti oluşturuluyor...")
        
        postinst_content = """#!/bin/bash
set -e

# Kullanıcı ve grup oluştur
if ! getent group wsanaliz > /dev/null 2>&1; then
    addgroup --system wsanaliz
fi

if ! getent passwd wsanaliz > /dev/null 2>&1; then
    adduser --system --ingroup wsanaliz --home /var/lib/wsanaliz --shell /bin/false wsanaliz
fi

# Dizin izinlerini ayarla
chown -R wsanaliz:wsanaliz /opt/wsanaliz
chown -R wsanaliz:wsanaliz /var/lib/wsanaliz
chown -R wsanaliz:wsanaliz /var/log/wsanaliz

# Python sanal ortamı oluştur
cd /opt/wsanaliz
sudo -u wsanaliz python3 -m venv venv
sudo -u wsanaliz ./venv/bin/pip install --upgrade pip
sudo -u wsanaliz ./venv/bin/pip install -r requirements.txt

# Veritabanını başlat
sudo -u wsanaliz ./venv/bin/python app.py --init-db

# Nginx sitesini etkinleştir
if [ -f /etc/nginx/sites-available/wsanaliz ]; then
    ln -sf /etc/nginx/sites-available/wsanaliz /etc/nginx/sites-enabled/
    nginx -t && systemctl reload nginx
fi

# Supervisor konfigürasyonunu yükle
if [ -f /etc/supervisor/conf.d/wsanaliz.conf ]; then
    supervisorctl reread
    supervisorctl update
    supervisorctl start wsanaliz
fi

# Systemd servisini etkinleştir
systemctl daemon-reload
systemctl enable wsanaliz
systemctl start wsanaliz

echo "✅ WSANALIZ başarıyla kuruldu!"
echo "🌐 Erişim: http://localhost"
echo "📊 Yönetim: http://localhost/admin"
"""
        
        postinst_path = self.package_dir / "DEBIAN" / "postinst"
        with open(postinst_path, "w", encoding="utf-8") as f:
            f.write(postinst_content)
        postinst_path.chmod(0o755)
        
    def create_prerm_script(self):
        """Kaldırma öncesi script oluştur"""
        print("🗑️ Pre-remove scripti oluşturuluyor...")
        
        prerm_content = """#!/bin/bash
set -e

# Servisleri durdur
systemctl stop wsanaliz || true
systemctl disable wsanaliz || true

# Supervisor'dan kaldır
supervisorctl stop wsanaliz || true

# Nginx sitesini devre dışı bırak
rm -f /etc/nginx/sites-enabled/wsanaliz
nginx -t && systemctl reload nginx || true

echo "🛑 WSANALIZ servisleri durduruldu"
"""
        
        prerm_path = self.package_dir / "DEBIAN" / "prerm"
        with open(prerm_path, "w", encoding="utf-8") as f:
            f.write(prerm_content)
        prerm_path.chmod(0o755)
        
    def create_nginx_config(self):
        """Nginx konfigürasyonu oluştur"""
        print("🌐 Nginx konfigürasyonu oluşturuluyor...")
        
        nginx_config = """server {
    listen 80;
    server_name localhost;
    
    # Güvenlik başlıkları
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Ana uygulama
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Upload boyutu
        client_max_body_size 100M;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # Statik dosyalar
    location /static {
        alias /opt/wsanaliz/app/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Loglar
    access_log /var/log/nginx/wsanaliz_access.log;
    error_log /var/log/nginx/wsanaliz_error.log;
}
"""
        
        with open(self.package_dir / "etc/nginx/sites-available/wsanaliz", "w") as f:
            f.write(nginx_config)
            
    def create_supervisor_config(self):
        """Supervisor konfigürasyonu oluştur"""
        print("👥 Supervisor konfigürasyonu oluşturuluyor...")
        
        supervisor_config = """[program:wsanaliz]
command=/opt/wsanaliz/venv/bin/python app.py
directory=/opt/wsanaliz
user=wsanaliz
group=wsanaliz
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/wsanaliz/app.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=FLASK_ENV=production,PYTHONPATH=/opt/wsanaliz
"""
        
        with open(self.package_dir / "etc/supervisor/conf.d/wsanaliz.conf", "w") as f:
            f.write(supervisor_config)
            
    def create_systemd_service(self):
        """Systemd service dosyası oluştur"""
        print("⚙️ Systemd service oluşturuluyor...")
        
        service_content = """[Unit]
Description=T.C. Aile ve Sosyal Hizmetler Bakanlığı İçerik Güvenlik Analiz Sistemi
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=wsanaliz
Group=wsanaliz
WorkingDirectory=/opt/wsanaliz
Environment=FLASK_ENV=production
Environment=PYTHONPATH=/opt/wsanaliz
ExecStart=/opt/wsanaliz/venv/bin/python app.py
Restart=always
RestartSec=10

# Güvenlik ayarları
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/wsanaliz /var/log/wsanaliz /opt/wsanaliz/storage

[Install]
WantedBy=multi-user.target
"""
        
        with open(self.package_dir / "etc/systemd/system/wsanaliz.service", "w") as f:
            f.write(service_content)
            
    def copy_application_files(self):
        """Uygulama dosyalarını kopyala"""
        print("📂 Uygulama dosyaları kopyalanıyor...")
        
        # Ana uygulama dizini
        app_dest = self.package_dir / "opt/wsanaliz"
        
        # Kopyalanacak dosya ve dizinler
        items_to_copy = [
            "app/",
            "storage/",
            "app.py",
            "config.py",
            "requirements.txt",
            "wsgi.py",
            "README.md",
            "LICENSE"
        ]
        
        for item in items_to_copy:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.copytree(item, app_dest / item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, app_dest / item)
                    
        # Executable script oluştur
        launcher_script = """#!/bin/bash
cd /opt/wsanaliz
exec ./venv/bin/python app.py "$@"
"""
        
        launcher_path = self.package_dir / "usr/bin/wsanaliz"
        with open(launcher_path, "w") as f:
            f.write(launcher_script)
        launcher_path.chmod(0o755)
        
    def build_package(self):
        """DEB paketini oluştur"""
        print("🔨 DEB paketi oluşturuluyor...")
        
        # dpkg-deb ile paketi oluştur
        package_name = f"{self.app_name}_{self.version}_{self.architecture}.deb"
        
        try:
            subprocess.run([
                "dpkg-deb", "--build", 
                str(self.package_dir), 
                package_name
            ], check=True)
            
            print(f"✅ DEB paketi oluşturuldu: {package_name}")
            
            # Paket bilgilerini göster
            subprocess.run(["dpkg-deb", "--info", package_name])
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Paket oluşturma hatası: {e}")
            return False
            
        return True
        
    def clean_build_dir(self):
        """Build dizinini temizle"""
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            
    def create_package(self):
        """Ana paket oluşturma fonksiyonu"""
        print("🚀 WSANALIZ DEB Paketi Oluşturuluyor...")
        print("=" * 50)
        
        try:
            # Önceki build'i temizle
            self.clean_build_dir()
            
            # Dizin yapısını oluştur
            self.create_directory_structure()
            
            # Kontrol dosyalarını oluştur
            self.create_control_file()
            self.create_postinst_script()
            self.create_prerm_script()
            
            # Konfigürasyon dosyalarını oluştur
            self.create_nginx_config()
            self.create_supervisor_config()
            self.create_systemd_service()
            
            # Uygulama dosyalarını kopyala
            self.copy_application_files()
            
            # Paketi oluştur
            success = self.build_package()
            
            if success:
                print("\n🎉 DEB paketi başarıyla oluşturuldu!")
                print("\n📦 Kurulum:")
                print(f"sudo dpkg -i {self.app_name}_{self.version}_{self.architecture}.deb")
                print("sudo apt-get install -f  # Bağımlılıkları çöz")
                print("\n🗑️ Kaldırma:")
                print(f"sudo apt remove {self.app_name}")
                
            return success
            
        except Exception as e:
            print(f"❌ Hata: {e}")
            return False

if __name__ == "__main__":
    builder = DebPackageBuilder()
    success = builder.create_package()
    sys.exit(0 if success else 1) 