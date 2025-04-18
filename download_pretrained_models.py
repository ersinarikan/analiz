import os
import json
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import shutil
from pathlib import Path

def create_base_model():
    """ContentAnalyzer'daki model yapısına uygun temel model oluşturur"""
    # MobileNetV2 tabanlı model
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Temel modeli dondur
    base_model.trainable = False
    
    # Üst katmanları ekle
    global_avg_layer = GlobalAveragePooling2D()
    
    # Model oluştur
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = global_avg_layer(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile et
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

def create_age_estimation_model():
    """Yaş tahmini için MobileNetV2 tabanlı model oluşturur"""
    # MobileNetV2 tabanlı model
    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # Yaş tahmini için regresyon
    ])
    
    # Compile et - MAE metrik ve loss fonksiyonu olarak doğrudan fonksiyon nesnelerini kullan
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    return model

def download_and_save_models(output_dir="storage/models", overwrite=False):
    """MobileNetV2 tabanlı temel modelleri oluşturup kaydeder"""
    model_types = ['violence', 'adult', 'harassment', 'weapon', 'drug']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for model_type in model_types:
        model_dir = os.path.join(output_dir, model_type)
        model_file = os.path.join(model_dir, 'model.h5')
        
        if os.path.exists(model_file) and not overwrite:
            print(f"{model_type} modeli zaten mevcut, atlanıyor (overwrite=False)")
            continue
        
        print(f"{model_type} modeli oluşturuluyor...")
        os.makedirs(model_dir, exist_ok=True)
        
        # Temel model oluştur
        model = create_base_model()
        
        # Modeli kaydet
        model.save(model_file, save_format='h5')
        print(f"{model_type} modeli başarıyla kaydedildi: {model_file}")
        
        # Modeli yüklemeyi dene (test amaçlı)
        try:
            loaded_model = tf.keras.models.load_model(model_file)
            print(f"Model doğrulama: {model_type} modeli başarıyla yüklendi")
        except Exception as e:
            print(f"HATA - {model_type} modeli yüklemesi başarısız: {str(e)}")
    
    # Yaş tahmini modeli oluştur
    age_model_dir = os.path.join(output_dir, 'age')
    age_model_file = os.path.join(age_model_dir, 'age_model.h5')
    
    if os.path.exists(age_model_file) and not overwrite:
        print(f"Yaş tahmini modeli zaten mevcut, atlanıyor (overwrite=False)")
    else:
        print("Yaş tahmini modeli oluşturuluyor...")
        os.makedirs(age_model_dir, exist_ok=True)
        
        # Yaş tahmini modeli oluştur
        age_model = create_age_estimation_model()
        
        # Modeli kaydet - HDF5 formatında
        age_model.save(age_model_file)
        print(f"Yaş tahmini modeli başarıyla kaydedildi: {age_model_file}")
        
        # Modeli yüklemeyi dene (test amaçlı)
        try:
            # Özel nesne sözlüğü ile yükleme
            custom_objects = {
                'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError
            }
            loaded_model = tf.keras.models.load_model(
                age_model_file, 
                custom_objects=custom_objects
            )
            print(f"Model doğrulama: Yaş tahmini modeli başarıyla yüklendi")
        except Exception as e:
            print(f"HATA - Yaş tahmini modeli yüklemesi başarısız: {str(e)}")
    
    print("Tüm modeller oluşturuldu!")

def download_yolo_model(output_dir="storage/models"):
    """YOLOv8 modelini indirir"""
    from ultralytics import YOLO
    
    detection_dir = os.path.join(output_dir, 'detection')
    os.makedirs(detection_dir, exist_ok=True)
    
    yolo_path = os.path.join(detection_dir, 'yolov8n.pt')
    
    if os.path.exists(yolo_path):
        print(f"YOLOv8 modeli zaten mevcut: {yolo_path}")
        return
    
    print("YOLOv8 modeli indiriliyor...")
    model = YOLO('yolov8n.pt')
    
    # İndirilen modeli kopyala
    if os.path.exists('yolov8n.pt'):
        shutil.copy('yolov8n.pt', yolo_path)
        print(f"YOLOv8 modeli başarıyla kopyalandı: {yolo_path}")
    else:
        print("HATA - YOLOv8 modeli indirilemedi!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorFlow modelleri indirme ve dönüştürme aracı')
    parser.add_argument('--output', default='storage/models', help='Modellerin kaydedileceği dizin')
    parser.add_argument('--overwrite', action='store_true', help='Mevcut modellerin üzerine yazılsın mı?')
    
    args = parser.parse_args()
    
    # Modelleri indir ve kaydet
    download_and_save_models(args.output, args.overwrite)
    download_yolo_model(args.output)
    
    print("İşlem tamamlandı!") 