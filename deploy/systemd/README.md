### Web + Worker (Redis Queue) kurulum notları

Bu repo artık queue’yu **Redis** üzerinden çalıştırabilecek şekilde düzenlendi.
Ama `/etc/systemd/system/*.service` dosyalarına yazmak için root yetkisi gerektiği için,
unit dosyaları repo içine `deploy/systemd/` altına eklendi.

## 1) Redis’in çalıştığını doğrula

`redis-server` servisiniz yoksa kurun/çalıştırın.

## 2) PAM ve unit dosyalarını kopyala

`unix_chkpwd` sadece çağıran kullanıcının şifresini doğruladığı için web (ersin) üzerinden serdar vb. giriş yapılamaz. Bunun için root’ta çalışan **wsanaliz-pamauth** ve `pam_rootok` içermeyen bir PAM servisi kullanılıyor.

```bash
# PAM: sadece pam_unix (pam_rootok yok)
sudo cp /opt/wsanaliz/deploy/pam/wsanaliz /etc/pam.d/wsanaliz

# systemd
sudo cp /opt/wsanaliz/deploy/systemd/wsanaliz-pamauth.service /etc/systemd/system/wsanaliz-pamauth.service
sudo cp /opt/wsanaliz/deploy/systemd/wsanaliz-web.service /etc/systemd/system/wsanaliz-web.service
sudo cp /opt/wsanaliz/deploy/systemd/wsanaliz-worker.service /etc/systemd/system/wsanaliz-worker.service
sudo systemctl daemon-reload
```

## 3) Eski servisi devre dışı bırak (opsiyonel ama önerilir)

```bash
sudo systemctl disable --now wsanaliz.service
```

## 4) Yeni servisleri başlat

```bash
sudo systemctl enable --now wsanaliz-pamauth.service
sudo systemctl enable --now wsanaliz-web.service
sudo systemctl enable --now wsanaliz-worker.service
```

## 5) Kontrol

```bash
sudo systemctl status wsanaliz-web.service --no-pager -l
sudo systemctl status wsanaliz-worker.service --no-pager -l
curl -sS http://127.0.0.1:5000/api/queue/stats | python3 -m json.tool
```

