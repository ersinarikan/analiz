# Bel2.rtf Devam Planı (3.7'den İtibaren)

**Tarih:** 30 Ekim 2025  
**Amaç:** Bel2.rtf'de 3.6'ya kadar tamamlanmış; 3.7'den itibaren mantıklı, tekrarsız, kod-doğrulanmış başlık hiyerarşisi ve içerik kılavuzu.

---

## ÖNEMLİ UYARI

**Bel2.rtf ile tez.md Arasındaki Temel Fark:**
- Bel2.rtf'de "BÖLÜM 3 SİSTEM ve MİMARİ" (3.1 Genel Mimari, 3.2 Yaşam Döngüsü, 3.3 Config, 3.4 Parametreler, 3.5 ER Modeli, 3.6 Platform)
- tez.md'de "BÖLÜM 3 GEREKSİNİMLER VE TASARIM İLKELERİ" (gereksinim, güvenlik, test, tasarım)
- **Bel2'nin mimari içeriği tez.md'de Bölüm 4–5'te yer alıyor.**

**Önerilen Yaklaşım:**
- Bel2.rtf referans kabul edilerek devam edilmeli.
- 3.7'den itibaren Bel2'nin Bölüm 3 altında mimari/teknik konular ilerlemeli.
- tez.md'deki "BÖLÜM 3 GEREKSİNİMLER" içeriği, Bel2'de 3.0 gibi kısa bir başlık altına özetlenebilir veya Bölüm 1'e entegre edilebilir.
- tez.md'deki Bölüm 4–10 içerikleri, Bel2'nin Bölüm 4–10'u olarak devam edebilir.

---

## Önerilen Bel2.rtf Devam Yapısı

### BÖLÜM 3 SİSTEM ve MİMARİ (devam)

**3.7. KATMANLI MİMARİ ve VERİ AKIŞI**
- **Kapsam:** Client/UI → Orta Katman (REST/WebSocket/Queue) → Arka Katman (AI servisleri) → Veri/Model Katmanı.
- **tez.md'den alınacak içerik:** 4.2 Katman Diyagramı (satır ~1856), 4.1.2 Modüler Yapı (satır ~1789).
- **Kod doğrulama:** `app/__init__.py` (modül yapısı), `app/routes/*`, `app/services/*`, `app/ai/*`.
- **Yeni yazılacak:** Katmanlar arası veri akışı (dosya yükleme → kuyruk → AI → sonuç → WebSocket).
- **Çıkarılacak tekrarlar:** 3.1'deki genel mimari ile çakışan cümleler; sadece katman detayı kalacak.
- **Tahmini sayfa:** +3–4 sayfa.

**3.8. SEQUENCE ve STATE DİYAGRAMLARI**
- **Kapsam:** Dosya yükleme/analiz başlatma, model sürüm aktivasyonu, analiz durum geçişleri.
- **tez.md'den alınacak içerik:** 4.3 Sequence Diyagramları (satır ~1863), 4.4 Durum Diyagramı (satır ~1917).
- **Kod doğrulama:** `app/routes/analysis_routes.py`, `app/services/queue_service.py`, `app/models/analysis.py` (status alanı).
- **Yeni yazılacak:** Diyagram açıklamaları (UML/Mermaid syntax); durum geçişleri (pending → running → completed/failed/cancelled).
- **Çıkarılacak tekrarlar:** 3.2'deki yaşam döngüsü ile örtüşen akış cümleleri.
- **Tahmini sayfa:** +2–3 sayfa.

**3.9. DAĞITIM MİMARİSİ ve ON-PREMISES KURULUM**
- **Kapsam:** On-prem topoloji, Nginx/reverse proxy, WSGI/HTTP, veritabanı, dosya sistemi; Windows/Linux kurulum adımları.
- **tez.md'den alınacak içerik:** 4.5 Dağıtım Diyagramı (satır ~1932), Ek K (satır ~4548).
- **Kod doğrulama:** `wsgi.py`, `start_production.sh`, `stop_production.sh`, `health_check.sh`, `requirements.txt`.
- **Yeni yazılacak:** Kurulum prosedürü (venv, pip, model indirme, .env, başlatma), systemd servis örneği, sağlık kontrolü.
- **Çıkarılacak tekrarlar:** 3.6'daki platform/container bölümü ile örtüşen Docker/container kısımları; sadece on-prem kurulum adımları kalacak.
- **Tahmini sayfa:** +3–4 sayfa.

---

### BÖLÜM 4 YAPAY ZEKÂ MODÜLLERİ ve EĞİTİM

**4.1. İÇERİK ANALİZİ (OpenCLIP)**
- **Kapsam:** CLIP model mimarisi (ViT-H, patch, attention), classification head, fine-tuning, prompt engineering, YOLO bağlamsal ayarlama, ensemble, safe score.
- **tez.md'den alınacak içerik:** 6.1 İçerik Analizi (satır ~3090–3287).
- **Kod doğrulama:** `app/ai/content_analyzer.py`.
- **Yeni yazılacak:** Kısa giriş (Bel2 2.2'deki CLIP teorisi ile bağlantı); mimari ve kod referansları esas.
- **Çıkarılacak tekrarlar:** 2.2'deki CLIP teorik kısmı ile örtüşen uzun açıklamalar; sadece uygulama detayı kalacak.
- **Tahmini sayfa:** +6–8 sayfa.

**4.2. YÜZ ve YAŞ TAHMİNİ (InsightFace + Özel Başlık)**
- **Kapsam:** Buffalo-L (RetinaFace, ArcFace, genderage), Custom Age Head mimarisi, model yönetimi, CLIP paylaşımı, çapraz sorgu, pseudo-label paketi, feedback döngüsü, hata toleransı.
- **tez.md'den alınacak içerik:** 6.2 Yüz ve Yaş Tahmini (satır ~3288–3574).
- **Kod doğrulama:** `app/ai/insightface_age_estimator.py`, `app/services/incremental_age_training_service_v2.py`.
- **Yeni yazılacak:** Kısa giriş (Bel2 2.2'deki InsightFace teorisi ile bağlantı).
- **Çıkarılacak tekrarlar:** 2.2'deki InsightFace teorik kısmı tekrarı.
- **Tahmini sayfa:** +7–9 sayfa.

**4.3. EĞİTİM DÖNGÜSÜ ve AĞIRLIKLI KAYIP**
- **Kapsam:** Ağırlıklı MSE kaybı, erken durdurma, IncrementalAgeModel (base + fine-tune branch, mix_weight), UTKFace ön-eğitimi, feedback verisi hazırlama, kişi bazlı tekilleştirme, embedding normalizasyonu.
- **tez.md'den alınacak içerik:** 6.3–6.4 (satır ~3486–3574).
- **Kod doğrulama:** `app/services/incremental_age_training_service_v2.py::train_incremental_model`, `app/scripts/train_custom_age_head.py`.
- **Yeni yazılacak:** Eğitim döngüsü akış diyagramı (opsiyonel).
- **Çıkarılacak tekrarlar:** 2.3'teki ağırlıklı kayıp teorik kısmı tekrarı.
- **Tahmini sayfa:** +4–5 sayfa.

---

### BÖLÜM 5 GÜVEN SKORU ve VERİ KALİTESİ

**5.1. ÇOK-MODELLİ UZLAŞI (Agreement)**
- **Kapsam:** Teorik formülasyon (üstel azalma), duyarlılık (sigma), alternatif metrikler, pratik çapraz sorgu.
- **tez.md'den alınacak içerik:** 7.1 Çok-Modelli Uzlaşı (satır ~3578–3641).
- **Kod doğrulama:** `app/ai/insightface_age_estimator.py::_calculate_confidence_with_clip` (çapraz sorgu mantığı).
- **Yeni yazılacak:** Kısa giriş (Bel2 2.3 ile bağlantı).
- **Çıkarılacak tekrarlar:** 2.3'teki model uzlaşısı teorik kısmı.
- **Tahmini sayfa:** +3–4 sayfa.

**5.2. CLIP TABANLI ANLAMSAL DOĞRULAMA**
- **Kapsam:** Kosinüs benzerliği, prompt engineering, CLIP güven skoru hesaplama, sigmoid normalizasyon, kod implementasyonu.
- **tez.md'den alınacak içerik:** 7.2 CLIP Tabanlı Anlamsal Doğrulama (satır ~3642–3748).
- **Kod doğrulama:** `app/ai/insightface_age_estimator.py::_calculate_confidence_with_clip`.
- **Yeni yazılacak:** Kısa giriş.
- **Çıkarılacak tekrarlar:** 2.3'teki CLIP doğrulama teorik kısmı.
- **Tahmini sayfa:** +3–4 sayfa.

**5.3. ROC/PR ANALİZİ ve EŞİK BELİRLEME**
- **Kapsam:** ROC/PR eğrileri, eşik tarama tablosu (T=0.5–0.9), üç aşamalı karar kuralı (≥0.75 yüksek, 0.5–0.75 orta, <0.5 düşük), demografik denge, tekrar kontrolü, denetim izi.
- **tez.md'den alınacak içerik:** 7.3 ROC/PR (satır ~3749–3899), Tablo 7.1 eşik tarama sonuçları.
- **Kod doğrulama:** Deneysel tablo (8K kontrol seti).
- **Yeni yazılacak:** Kısa giriş, eşik seçim gerekçesi özeti.
- **Çıkarılacak tekrarlar:** 2.3'teki ROC/PR teorik kısmı.
- **Tahmini sayfa:** +4–5 sayfa.

**5.4. KAYIT, İZLENEBİLİRLİK ve YANLILIK DENETİMLERİ**
- **Kapsam:** Audit log yapısı, sürüm notları, demografik denge izleme, feedback döngüsü faydaları.
- **tez.md'den alınacak içerik:** 7.4–7.7 (satır ~3900–3929).
- **Kod doğrulama:** `app/models/feedback.py`, `app/models/analysis.py` (metadata alanları).
- **Yeni yazılacak:** Kısa özet.
- **Çıkarılacak tekrarlar:** Yok (bu kısım Bel2'de yok).
- **Tahmini sayfa:** +2–3 sayfa.

---

### BÖLÜM 6 ARKA UÇ UYGULAMASI

**6.1. REST API DETAYLARI**
- **Kapsam:** Endpoint kategorileri, dosya yükleme API, analiz başlatma API, görsel servis ve yol güvenliği.
- **tez.md'den alınacak içerik:** 5.1 REST API (satır ~2523–2783).
- **Kod doğrulama:** `app/routes/file_routes.py`, `app/routes/analysis_routes.py`.
- **Yeni yazılacak:** Kısa giriş (Bel2 3.1 genel mimari ile bağlantı).
- **Çıkarılacak tekrarlar:** 3.1 ve 3.6'daki API açıklamaları ile örtüşenler.
- **Tahmini sayfa:** +4–5 sayfa.

**6.2. WEBSOCKET ve GERÇEKZAMANLI BİLDİRİMLER**
- **Kapsam:** RFC 6455, Socket.IO konfigürasyonu, event types, room-based broadcasting, ping/pong, error handling, reconnection, monitoring.
- **tez.md'den alınacak içerik:** 5.2–5.3 WebSocket (satır ~2808–3031).
- **Kod doğrulama:** `app/socketio_instance.py`, `app/routes/websocket_routes.py`.
- **Yeni yazılacak:** Kısa giriş (Bel2 3.2 yaşam döngüsü ile bağlantı).
- **Çıkarılacak tekrarlar:** 3.2'deki Socket.IO açıklamaları.
- **Tahmini sayfa:** +5–6 sayfa.

**6.3. GÜVENLİK ve ARA BİLEŞEN (Middleware)**
- **Kapsam:** Oran sınırlama, güvenlik başlıkları (CSP, HSTS), MIME/magic-bytes doğrulama, path traversal koruması, OWASP Top 10.
- **tez.md'den alınacak içerik:** 5.4 Güvenlik (satır ~3032–3048), 3.12 Güvenlik ve Uyumluluk (satır ~1781–1789).
- **Kod doğrulama:** `app/middleware/security_middleware.py`, `app/utils/file_utils.py`, `app/utils/path_utils.py`.
- **Yeni yazılacak:** Kısa giriş (Bel2 3.6 platform güvenlik ile bağlantı).
- **Çıkarılacak tekrarlar:** tez.md 3.12 ve 5.4 tekrarları; tek bir yerden anlatım.
- **Tahmini sayfa:** +3–4 sayfa.

**6.4. KUYRUK İŞLEME ve HATA YÖNETİMİ**
- **Kapsam:** Kuyruk ekleme (FIFO), analiz dosya türü seçimi, yaş tahmini executor, ilerleme güncellemeleri, rollback, asenkron yaş tahmini, riskli kare kırpma.
- **tez.md'den alınacak içerik:** 5.2 Servis İş Akışı ve Kuyruk (satır ~2784–2807), 5.5 Hata Yönetimi (satır ~3052–3066).
- **Kod doğrulama:** `app/services/queue_service.py`, `app/services/analysis_service.py`.
- **Yeni yazılacak:** Kısa giriş (Bel2 3.2 yaşam döngüsü ile bağlantı).
- **Çıkarılacak tekrarlar:** 3.2'deki kuyruk açıklamaları.
- **Tahmini sayfa:** +4–5 sayfa.

**6.5. VİDEO ve GÖRSEL AKIŞ FARKLILIKLARI**
- **Kapsam:** Görsel akışı (tek kare), video akışı (kare örnekleme, DeepSORT, PersonTracker), overlay üretimi, kişi bazlı en güvenilir kare seçimi.
- **tez.md'den alınacak içerik:** 5.7 Video/Görsel Farklar (satır ~3067–3069), 5.8 Kişi Takibi (satır ~3070–3084).
- **Kod doğrulama:** `app/services/analysis_service.py`, `app/utils/person_tracker.py`, `app/utils/image_utils.py`.
- **Yeni yazılacak:** Akış karşılaştırması (tablo: görsel vs. video; süre, işlem, overlay).
- **Çıkarılacak tekrarlar:** Bel2 2.4'teki kişi takibi ile örtüşenler.
- **Tahmini sayfa:** +3–4 sayfa.

**6.6. 18 YAŞ ALTI TESPİT ve SOSYAL KORUMA UYARILARI**
- **Kapsam:** 18 yaş altı + yüksek risk (violence/adult >0.6) → kırmızı overlay, uyarı metni, special_warnings flag, öncelikli sıra, istatistik özeti.
- **tez.md'den alınacak içerik:** 5.10 18 Yaş Altı Tespit (satır ~3085–3087).
- **Kod doğrulama:** `app/services/analysis_service.py` (overlay üretimi, uyarı mantığı).
- **Yeni yazılacak:** Türkiye Cumhuriyeti Aile ve Sosyal Hizmetler Bakanlığı mevzuatı atfı (opsiyonel; Bel2 1.1'de geçiyor).
- **Çıkarılacak tekrarlar:** Yok (bu kısım Bel2'de yok).
- **Tahmini sayfa:** +2 sayfa.

---

### BÖLÜM 7 DENEYSEL KURULUM ve SONUÇLAR

**7.1. ORTAM ve VERİ**
- **Kapsam:** Donanım/yazılım spesifikasyonları (GPU, CUDA, Python, PyTorch, OpenCLIP, YOLO, InsightFace, Flask), üretim verisi (450 analiz, 180K dosya, 303K yüz, 8K kontrol seti, 5 eğitim), model sürümleri, tohumlar, hiperparametreler.
- **tez.md'den alınacak içerik:** 8.1 Ortam ve Veri (satır ~3938–3965).
- **Kod doğrulama:** `requirements.txt`, `config.py`.
- **Yeni yazılacak:** Kısa giriş (gerçek üretim verisi vurgusu).
- **Çıkarılacak tekrarlar:** Yok (bu kısım Bel2'de yok).
- **Tahmini sayfa:** +3–4 sayfa.

**7.2. ÖLÇÜTLER ve PROTOKOL**
- **Kapsam:** Performans metrikleri (MAE, MSE, ±3/±5y accuracy; precision, recall, F1, accuracy), hesaplama verimliliği (süre, bellek, throughput), çoğaltılabilirlik (tohumlar, K-fold CV, eşik kalibrasyonu).
- **tez.md'den alınacak içerik:** 8.2 Ölçütler (satır ~3966–3990).
- **Kod doğrulama:** `app/services/age_training_service.py::calculate_metrics`.
- **Yeni yazılacak:** Kısa giriş.
- **Çıkarılacak tekrarlar:** Bel2 2.3'teki metrik tanımları tekrarı (sadece atıf).
- **Tahmini sayfa:** +2–3 sayfa.

**7.3. ABLATION ÇALIŞMASI (C1–C5)**
- **Kapsam:** Deneysel kurulumlar (C1 tüm veri, C2 hafif başlık tüm veri, C3 hafif başlık güven filtreli, C4 C3+ağırlıklı kayıp, C5 tek bileşen ablation), Tablo 8.1 sonuçları.
- **tez.md'den alınacak içerik:** 8.3 Ablation (satır ~3992–4033), Tablo 8.1.
- **Kod doğrulama:** 8K kontrol seti (üretim verisi).
- **Yeni yazılacak:** Kısa giriş, tablo yorumu.
- **Çıkarılacak tekrarlar:** Yok (bu kısım Bel2'de yok).
- **Tahmini sayfa:** +3–4 sayfa.

**7.4. BULGULAR ve TARTIŞMA**
- **Kapsam:** Ana bulgular (C4 en iyi: MAE 5.7±0.2, ±5y %88.3), veri kalitesi filtreleme etkisi, ağırlıklı kayıp etkisi, çoklu-bileşen analizi, eşik duyarlılığı, yaş bandı hata profili, kaynak kullanımı, sürümleme etkisi.
- **tez.md'den alınacak içerik:** 8.4 Bulgular (satır ~4034–4077).
- **Kod doğrulama:** Deneysel sonuçlar (8K kontrol seti).
- **Yeni yazılacak:** Kısa giriş.
- **Çıkarılacak tekrarlar:** Yok.
- **Tahmini sayfa:** +3–4 sayfa.

**7.5. GEÇERLİLİK TEHDİTLERİ ve KARŞILAŞTIRMA**
- **Kapsam:** İç geçerlilik (örneklem bias, veri kayması, pseudo-label gürültüsü), dış geçerlilik (domain sınırlılığı, temporal bias), ölçüm güvenilirliği, alternatif sistemlerle karşılaştırma (Google Vision, AWS Rekognition, NudeNet) Tablo 8.2.
- **tez.md'den alınacak içerik:** 8.5 Geçerlilik Tehditleri (satır ~4078–4116), Tablo 8.2.
- **Kod doğrulama:** Karşılaştırma verileri (100 test görseli ile).
- **Yeni yazılacak:** Kısa giriş.
- **Çıkarılacak tekrarlar:** Yok.
- **Tahmini sayfa:** +3–4 sayfa.

**7.6. UYGULAMA İMPLİKASYONLARI**
- **Kapsam:** Performans/kaynak optimizasyonu (GPU, işlem süresi, kişi takibi), kullanıcı deneyimi (overlay, çoklu dosya, riskli kare kırpma), veri yönetimi (cleanup, DB), güvenlik/gizlilik (on-prem, KVKK), model yönetimi (sürümleme, rollback, otomatik etiketleme), operasyonel faydalar (maliyet, bakım, ölçeklenebilirlik).
- **tez.md'den alınacak içerik:** 8.6 Uygulama İmplikasyonları (satır ~4117–4162).
- **Kod doğrulama:** Üretim verisi (450 analiz, 180K dosya, 303K yüz).
- **Yeni yazılacak:** Kısa giriş.
- **Çıkarılacak tekrarlar:** Bel2 1.3'teki katkılar ile örtüşenler (sadece atıf).
- **Tahmini sayfa:** +3–4 sayfa.

---

### BÖLÜM 8 ARKA UÇ UYGULAMASI (Detaylı Akış)

**Not:** Bu bölüm, tez.md'deki Bölüm 5'in daha ayrıntılı versiyonu olabilir. Alternatif olarak Bölüm 6'ya birleştirilebilir veya Bölüm 5'in alt başlıkları olarak düzenlenebilir. Bel2 akışında Bölüm 6 "Arka Uç" ise, aşağıdaki yapı önerilir:

**8.1. UÇTAN UCA İŞ AKIŞI**
- **tez.md'den:** 5.0 Uçtan Uca İş Akışı (satır ~2489–2522).
- **Kod:** `app/services/analysis_service.py`.
- **Tahmini sayfa:** +2 sayfa.

**8.2. SERVİS İŞ AKIŞI ve KUYRUK**
- **tez.md'den:** 5.2 Servis İş Akışı (satır ~2784–2807).
- **Kod:** `app/services/queue_service.py`.
- **Tahmini sayfa:** +2 sayfa.

**8.3. HATA YÖNETİMİ, MİGRATION ve TEMİZLİK**
- **tez.md'den:** 5.5 Hata Yönetimi (satır ~3052–3066).
- **Kod:** `app/database.py` (migration), `app/services/analysis_service.py` (cleanup).
- **Tahmini sayfa:** +2 sayfa.

---

### BÖLÜM 9 SONUÇ ve GELECEK ÇALIŞMALAR

- **Kapsam:** Özet (katkılar: çok-modelli güven, on-prem artımsal, kişi takibi+içerik+yaş, ROC eşik optimizasyonu), sınırlılıklar (domain sınırlılığı, Türkçe veri azlığı, GPU gereksinimi), gelecek çalışmalar (streaming video, aktif öğrenme, cross-domain validasyon, adversarial robustness, explainability, ensemble genişletme).
- **tez.md'den alınacak içerik:** Bölüm 9 (satır ~4163–4172).
- **Kod doğrulama:** Yok (sentez bölümü).
- **Yeni yazılacak:** Kısa ve net sentez; Bel2 1.3 ve 2.4 ile tutarlı.
- **Çıkarılacak tekrarlar:** 1.3 katkılar ile örtüşenler (sadece özet atıf).
- **Tahmini sayfa:** +2–3 sayfa.

---

### BÖLÜM 10 KULLANICI ARAYÜZÜ ve EKRANLAR

- **Kapsam:** Ana sayfa, dosya yükleme, yakın zamanlı analizler, detay modalı (tablar), model yönetimi, parametreler, yardım, WebSocket durumu, görsel büyütme.
- **tez.md'den alınacak içerik:** Bölüm 10 (satır ~4173–4254).
- **Kod doğrulama:** `app/templates/index.html`, `app/static/js/*.js`, `app/static/css/*.css`.
- **Ekran görüntüleri:** `app/static/img/ui/home_overview.png`, `upload_section.png`, `recent_analyses_cards.png`, `detail_modal_general.png`, `detail_modal_content.png`, `detail_modal_age.png`, `detail_modal_feedback.png`, `model_management_active.png`, `model_versions_history.png`, `training_params_modal.png`, `settings_modal.png`, `help_modal.png`, `websocket_status.png`, `toast_notifications.png`, `image_zoom_overlay.png`.
- **Yeni yazılacak:** Kısa giriş (Bel2 3.6 ile bağlantı: "platform katmanında kullanıcı arayüzü").
- **Çıkarılacak tekrarlar:** Yok (bu kısım Bel2'de yok).
- **Tahmini sayfa:** +5–6 sayfa (ekran görüntüleri ile).

---

### EKLER

- **Ek A–I:** Kod alıntıları (dosya yükleme, analiz başlatma, kuyruk, Socket.IO, güvenlik, migration, CLIP paylaşımı, sürüm kaydı, overlay, tekilleştirme, UTKFace eğitimi, bağlamsal ayarlama).
- **Ek K:** Windows/Linux kurulum adımları.
- **Ek L (opsiyonel):** Kapsamlı kod listesi (tam fonksiyonlar).
- **Tahmini sayfa:** +8–10 sayfa.

---

## Tekrar Risk Haritası

| Konu | Bel2.rtf | tez.md | Tekrar Riski | Nasıl Önlenir |
|------|----------|--------|--------------|---------------|
| CLIP teorisi | 2.2 kısa | 2.2.1 + 6.1 detaylı | Orta | 2.2'de teorik, 4.1'de (önerilen bölüm numarası) uygulama; sadece atıf |
| Güven skorlama | 2.3 teorik | 2.3 + 5.x (önerilen) + 7.x | Yüksek | 2.3'te teorik, 5.x'te uygulama, 7.x'te deneysel; her yerde farklı açı |
| Platform bağımsızlık | 3.6 | 3.8–3.9 + Ek K | Orta | 3.6'da ilke, Ek K'da kurulum; metin kısa atıf |
| Kişi takibi | 2.4 kısa | 5.8 + 6.5 (önerilen) | Orta | 2.4'te literatür boşluğu, 6.5'te uygulama; atıf ile bağla |
| ROC/PR | 2.3 kısa | 5.3 (önerilen) + 7.3 detaylı | Orta | 2.3'te teorik, 5.3'te eşik seçimi, 7.3'te deneysel tablo; atıf |
| Model sürümleme | 3.3–3.4 kısmi | 6.2 + Ek A.8 | Düşük | 3.3–3.4'te config/durum, 6.2'de kod; kısa atıf |

---

## Sayfa Hedefi Dağılımı

| Bölüm | Tahmini Eklenen Sayfa | Toplam Tahmini |
|-------|----------------------|----------------|
| Bölüm 1–3.6 (Bel2 mevcut) | 0 (korunuyor) | ~37 sayfa |
| Bölüm 3.7–3.9 (mimari devam) | +8–10 | ~45–47 |
| Bölüm 4 (YZ Modülleri) | +17–22 | ~62–69 |
| Bölüm 5 (Güven Skoru) | +12–16 | ~74–85 |
| Bölüm 6 (Arka Uç) | +15–20 | ~89–105 |
| Bölüm 7 (Deneysel) | +18–22 | ~107–127 |
| Bölüm 8 (Arka Uç Detay, opsiyonel) | +6–8 (veya Bölüm 6'ya birleştir) | ~113–135 |
| Bölüm 9 (Sonuç) | +2–3 | ~115–138 |
| Bölüm 10 (UI) | +5–6 | ~120–144 |
| Ekler | +8–10 | ~128–154 |

**Toplam Hedef:** ≈128–154 sayfa (ekler dahil); **ekler hariç ≈120–144 sayfa**.

**Hedef ≈150 sayfa (ekler hariç) için:** Her bölümde 1–2 sayfa daha genişletme yapılabilir veya Bölüm 8'i ayrı tutarak toplam 6–8 sayfa eklenebilir.

---

## Sonuç ve Tavsiyeler

### Öncelikli Aksiyonlar

1. **Bölüm 3 Yapısını Netleştirin:**
   - Bel2.rtf'de "BÖLÜM 3 SİSTEM ve MİMARİ" olarak devam edin.
   - tez.md'deki "BÖLÜM 3 GEREKSİNİMLER" içeriğini başka yere taşıyın veya Bel2'nin Bölüm 1 veya 2'sine entegre edin.
   - Bel2'nin 3.1–3.6'sını aynen koruyun.

2. **3.7–3.9 Ekleyin (Mimari Devam):**
   - Katmanlı mimari, sequence/state diyagramları, dağıtım/kurulum.
   - tez.md'nin 4.2–4.5 ve Ek K'sından alın.

3. **Bölüm 4–10 Ekleyin:**
   - YZ Modülleri (tez 6.x), Güven Skoru (tez 7.x), Deneysel (tez 8.x), Sonuç (tez 9), UI (tez 10).
   - Her bölümde Bel2 2.x ile örtüşen kısımlara sadece atıf yapın (tekrar etmeyin).

4. **Ekler ve Referansları Birleştirin:**
   - tez.md'deki Ek A–K'yi ekleyin.
   - KAYNAKLAR'ı Bel2 ile birleştirin (şu an zaten uyumlu).

5. **Dil ve Akış Kontrolü:**
   - Bel2 üslubu korunarak (akademik, kod-doğrulamalı, bütünlüklü).
   - Tekrar cümleleri temizleyin; her konuyu tek bir bölümde detaylı anlatın, diğer yerlerde kısa atıf yapın.

### Nihai Hedef

- **Bel2.rtf (3.6'ya kadar) + Önerilen 3.7–10 + Ekler = ≈150 sayfa (ekler hariç ≈140)**
- Akademik üslup, kod doğrulamalı, tekrarsız, bütünlüklü.

