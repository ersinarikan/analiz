# Settings State File
# Bu dosya parametre değişikliklerinde güncellenir ve Flask debug mode tarafından izlenir
# Otomatik restart için config.py tarafından import edilir

SETTINGS_STATE = {
    'face_detection_confidence': 0.2,
    'tracking_reliability_threshold': 0.3,
    'id_change_threshold': 0.45,
    'max_lost_frames': None,
    'embedding_distance_threshold': 0.3
}

# Bu satır Flask'ın dosya değişikliklerini algılaması için
# Her parametre değişikliğinde timestamp güncellenir
LAST_UPDATE = "2025-07-31T13:31:00.974626"