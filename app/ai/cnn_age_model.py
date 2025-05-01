import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

def create_age_model():
    """
    Yaş tahmini için CNN modeli oluşturur.
    ResNet benzeri bir mimari kullanır.
    """
    model = models.Sequential([
        # Giriş katmanı
        layers.Input(shape=(224, 224, 3)),
        
        # İlk konvolüsyon bloğu
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # İkinci konvolüsyon bloğu
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Üçüncü konvolüsyon bloğu
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dördüncü konvolüsyon bloğu
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Yoğun katmanlar
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='linear')  # Yaş tahmini için tek çıkış
    ])
    
    return model

def train_model(model, train_data, val_data, epochs=100, batch_size=32, learning_rate=0.001):
    """
    Modeli eğitir.
    
    Args:
        model: Eğitilecek model
        train_data: (X_train, y_train) tuple
        val_data: (X_val, y_val) tuple
        epochs: Eğitim epoch sayısı
        batch_size: Batch boyutu
        learning_rate: Öğrenme oranı
    """
    try:
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Optimizer
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # Model derleme
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Ortalama kare hata
            metrics=['mae']  # Ortalama mutlak hata
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Model eğitimi
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Model eğitim hatası: {str(e)}")
        raise

def save_model(model, path):
    """Modeli kaydeder"""
    try:
        model.save(path)
        logger.info(f"Model başarıyla kaydedildi: {path}")
    except Exception as e:
        logger.error(f"Model kaydetme hatası: {str(e)}")
        raise 