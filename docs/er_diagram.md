# WSANALIZ Veritabanı ER Diagramı

## Entity Relationship Diagram

```mermaid
erDiagram
    FILES {
        int id PK
        string filename
        string original_filename
        string file_path UK
        int file_size
        string mime_type
        string file_type
        datetime created_at
        int user_id FK
    }
    
    ANALYSES {
        string id PK
        string file_id FK
        string status
        datetime start_time
        datetime end_time
        datetime created_at
        int frames_analyzed
        float frames_per_second
        text error_message
        boolean include_age_analysis
        string websocket_session_id
        boolean is_cancelled
        float overall_violence_score
        float overall_adult_content_score
        float overall_harassment_score
        float overall_weapon_score
        float overall_drug_score
        float overall_safe_score
        string highest_risk_frame
        float highest_risk_frame_timestamp
        float highest_risk_score
        string highest_risk_category
        text category_specific_highest_risks_data
    }
    
    CONTENT_DETECTIONS {
        int id PK
        string analysis_id FK
        string frame_path
        float frame_timestamp
        int frame_index
        float violence_score
        float adult_content_score
        float harassment_score
        float weapon_score
        float drug_score
        float safe_score
        text detected_objects
    }
    
    AGE_ESTIMATIONS {
        int id PK
        string analysis_id FK
        string person_id
        string frame_path
        float frame_timestamp
        int frame_index
        float estimated_age
        float confidence_score
        string face_bbox
        text face_landmarks
        text embedding
        string age_range
        string processed_image_path
    }
    
    FEEDBACK {
        int id PK
        datetime created_at
        string frame_path
        string face_bbox
        text embedding
        string content_id
        string analysis_id
        string person_id
        int corrected_age
        float pseudo_label_original_age
        float pseudo_label_clip_confidence
        boolean is_age_range_correct
        string feedback_type
        string feedback_source
        int rating
        text comment
        json category_feedback
        json category_correct_values
        string training_status
        string used_in_model_version
        datetime training_used_at
        boolean is_archived
        string archive_reason
        boolean used_in_ensemble
        int ensemble_usage_count
        datetime last_used_at
        json ensemble_model_versions
    }
    
    MODEL_VERSIONS {
        int id PK
        string model_type
        int version
        string version_name
        datetime created_at
        json metrics
        boolean is_active
        int training_samples
        int validation_samples
        int epochs
        string file_path
        string weights_path
        json used_feedback_ids
    }
    
    %% İlişkiler
    FILES ||--o{ ANALYSES : "has many"
    ANALYSES ||--o{ CONTENT_DETECTIONS : "contains"
    ANALYSES ||--o{ AGE_ESTIMATIONS : "contains"
    ANALYSES ||--o{ FEEDBACK : "receives"
    MODEL_VERSIONS ||--o{ FEEDBACK : "uses for training"
```

## Tablo Açıklamaları

### FILES (Dosyalar)
- Yüklenen dosyaların temel bilgilerini saklar
- Her dosya birden fazla analize sahip olabilir
- Dosya türü (image/video) ve MIME tipi bilgileri

### ANALYSES (Analizler)
- Her dosya için yapılan analiz işlemlerini temsil eder
- Analiz durumu, zaman bilgileri ve genel skorları içerir
- WebSocket session takibi ve iptal mekanizması

### CONTENT_DETECTIONS (İçerik Tespitleri)
- Her kare için içerik analiz sonuçlarını saklar
- 5 risk kategorisi skorları (violence, adult_content, harassment, weapon, drug)
- Tespit edilen nesneler JSON formatında

### AGE_ESTIMATIONS (Yaş Tahminleri)
- Yüz tespit edilen her kişi için yaş tahmin sonuçları
- Kişi takibi için embedding ve person_id
- Yüz bounding box ve landmark bilgileri

### FEEDBACK (Geri Bildirimler)
- Kullanıcı geri bildirimleri ve sözde etiketler
- Eğitim sürecinde kullanım takibi
- Ensemble mekanizması için kullanım geçmişi

### MODEL_VERSIONS (Model Sürümleri)
- Eğitilmiş modellerin sürüm bilgileri
- Performans metrikleri ve eğitim parametreleri
- Hangi geri bildirimlerin kullanıldığı bilgisi

## İlişki Türleri

- **1:N** - Bir dosya birden fazla analize sahip olabilir
- **1:N** - Bir analiz birden fazla içerik tespiti içerebilir
- **1:N** - Bir analiz birden fazla yaş tahmini içerebilir
- **1:N** - Bir analiz birden fazla geri bildirim alabilir
- **1:N** - Bir model sürümü birden fazla geri bildirim kullanabilir

## Önemli Özellikler

1. **UUID Kullanımı**: Analysis ve person_id alanlarında UUID kullanımı
2. **JSON Alanlar**: Esnek veri saklama için JSON alanları
3. **İndeksleme**: Performans için kritik alanlarda indeksler
4. **Cascade Delete**: İlişkili kayıtların otomatik silinmesi
5. **Audit Trail**: Oluşturulma ve güncellenme zamanları

