# ERSIN Settings State File
# ERSIN Bu dosya parametre değişikliklerinde güncellenir ve Flask debug mode tarafından izlenir
# ERSIN Otomatik restart için config.py tarafından import edilir

SETTINGS_STATE ={
'face_detection_confidence':0.25 ,
'tracking_reliability_threshold':0.5 ,
'id_change_threshold':0.55 ,
'max_lost_frames':None ,
'embedding_distance_threshold':0.4 
}

# ERSIN Bu satır Flask'ın dosya değişikliklerini algılaması için
# ERSIN Her parametre değişikliğinde timestamp güncellenir
LAST_UPDATE ="2026-01-27T20:07:01.302990"