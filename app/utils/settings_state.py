# Settings State File
# Bu dosya parametre değişikliklerinde güncellenir ve Flask debug mode tarafından izlenir
# Otomatik restart için config.py tarafından import edilir

SETTINGS_STATE = {
    'face_detection_confidence': 0.2,
    'tracking_reliability_threshold': 0.35,
    'id_change_threshold': 0.55,
    'max_lost_frames': 30,
    'embedding_distance_threshold': 0.45
}

# Bu satır Flask'ın dosya değişikliklerini algılaması için
# Her parametre değişikliğinde timestamp güncellenir
LAST_UPDATE = "2025-05-27T18:09:36.907468"