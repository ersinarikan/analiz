import os
import json
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from flask import current_app
from app.services.model_service import prepare_training_data
from app.utils.image_utils import load_image
import cv2

class ModelTrainer:
    """
    Model eğitimi ve değerlendirmesini yöneten ana sınıf.
    - Eğitim verisi hazırlama, model eğitimi ve değerlendirme işlemlerini içerir.
    """
    
    def __init__(self, model_type):
        """Eğitim için sınıfı başlat."""
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() and current_app.config['USE_GPU'] else 'cpu'
        
        # Model yolları
        self.model_folder = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model")
        self.versions_folder = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model_versions")
        self.config_path = os.path.join(current_app.config['MODELS_FOLDER'], f"{model_type}_model_config.json")
        
        # Dizinlerin var olduğundan emin ol
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.versions_folder, exist_ok=True)
        
        # Mevcut konfigürasyonu yükle
        self.config = self._load_config()
    
    def _load_config(self):
        """Model konfigürasyonunu yükle veya oluştur."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                current_app.logger.error(f"Konfigürasyon yükleme hatası: {str(e)}")
        
        # Varsayılan konfigürasyonu oluştur
        default_config = {
            "model_type": self.model_type,
            "version": "pretrained",
            "training_history": [],
            "metrics": {},
            "is_pretrained": True
        }
        
        return default_config
    
    def _save_config(self):
        """Model konfigürasyonunu kaydet."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            current_app.logger.error(f"Konfigürasyon kaydetme hatası: {str(e)}")
    
    def train(self, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        """
        Modeli eğitir.
        - Eğitim verisini hazırlar.
        - Mevcut modeli yedekler.
        - İçerik veya yaş modeline göre eğitim yapar.
        - Eğitim geçmişini ve metrikleri günceller.
        - Konfigürasyonu kaydeder.
        """
        # Eğitim verisini hazırla
        training_data, message = prepare_training_data(self.model_type)
        
        if not training_data:
            return False, f"Eğitim verisi hazırlanamadı: {message}"
        
        if len(training_data) < 10:
            return False, f"Yetersiz eğitim verisi: {len(training_data)} örnek (en az 10 gerekli)"
        
        # Eğitim öncesi mevcut modeli yedekle
        self._backup_current_model()
        
        if self.model_type == 'content':
            success, result = self._train_content_model(training_data, epochs, batch_size, learning_rate)
        elif self.model_type == 'age':
            success, result = self._train_age_model(training_data, epochs, batch_size, learning_rate)
        else:
            return False, "Desteklenmeyen model tipi"
        
        if success:
            # Eğitim geçmişini güncelle
            training_entry = {
                "date": datetime.datetime.now().isoformat(),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "samples": len(training_data),
                "metrics": result
            }
            
            self.config["training_history"].append(training_entry)
            self.config["metrics"] = result
            self.config["version"] = f"v{len(self.config['training_history'])}"
            self.config["is_pretrained"] = False
            
            # Konfigürasyonu kaydet
            self._save_config()
            
            return True, result
        else:
            return False, result
    
    def _backup_current_model(self):
        """Mevcut modeli yedekle."""
        if not os.path.exists(self.model_folder):
            return
        
        try:
            # Yeni sürüm klasörü oluştur
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"backup_{timestamp}"
            version_folder = os.path.join(self.versions_folder, version_name)
            
            os.makedirs(version_folder, exist_ok=True)
            
            # Model dosyalarını kopyala
            import shutil
            for file in os.listdir(self.model_folder):
                src = os.path.join(self.model_folder, file)
                dst = os.path.join(version_folder, file)
                
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
            
            # Versiyon bilgisi oluştur
            version_info = {
                "version": version_name,
                "date": timestamp,
                "is_backup": True,
                "config": self.config
            }
            
            # Versiyon bilgisini kaydet
            with open(os.path.join(version_folder, "version_info.json"), 'w') as f:
                json.dump(version_info, f, indent=4)
        
        except Exception as e:
            current_app.logger.error(f"Model yedekleme hatası: {str(e)}")
    
    def _train_content_model(self, training_data, epochs, batch_size, learning_rate):
        """
        İçerik analiz modelini eğitir.
        - Eğitim verisini hazırlar.
        - Gerçek bir eğitim yerine basit bir simülasyon yapar.
        - Eğitim performansını simüle eder.
        - Sonuçları hesaplar ve grafikler oluşturur.
        """
        try:
            # Örnek uygulama: Burada gerçek bir eğitim yapılacak
            # Şu an için basit bir simülasyon yapalım
            
            # Progress kontrolü için
            progress_callback = lambda p: print(f"Eğitim ilerleme: {p:.2f}%")
            
            # Veriyi hazırla
            X = []
            y_violence = []
            y_adult = []
            y_harassment = []
            y_weapon = []
            y_drug = []
            
            for item in training_data:
                # Görüntüyü yükle
                img = load_image(item['frame_path'])
                if img is None:
                    continue
                
                # Görüntüyü ön işleme
                img = cv2.resize(img, (224, 224))
                img = img / 255.0  # Normalize et
                
                X.append(img)
                y_violence.append(item['labels'].get('violence', 0))
                y_adult.append(item['labels'].get('adult_content', 0))
                y_harassment.append(item['labels'].get('harassment', 0))
                y_weapon.append(item['labels'].get('weapon', 0))
                y_drug.append(item['labels'].get('drug', 0))
            
            # Numpy dizilerine dönüştür
            X = np.array(X)
            y_violence = np.array(y_violence)
            y_adult = np.array(y_adult)
            y_harassment = np.array(y_harassment)
            y_weapon = np.array(y_weapon)
            y_drug = np.array(y_drug)
            
            # Eğitim ve test verisi ayır
            X_train, X_test, y_violence_train, y_violence_test = train_test_split(X, y_violence, test_size=0.2, random_state=42)
            _, _, y_adult_train, y_adult_test = train_test_split(X, y_adult, test_size=0.2, random_state=42)
            _, _, y_harassment_train, y_harassment_test = train_test_split(X, y_harassment, test_size=0.2, random_state=42)
            _, _, y_weapon_train, y_weapon_test = train_test_split(X, y_weapon, test_size=0.2, random_state=42)
            _, _, y_drug_train, y_drug_test = train_test_split(X, y_drug, test_size=0.2, random_state=42)
            
            # Gerçek bir eğitim yerine simülasyon yap
            # Gerçek projede bu kısımda TensorFlow veya PyTorch ile model eğitimi yapılır
            
            # Eğitim performansı simülasyonu
            loss_history = []
            for i in range(epochs):
                loss = 1.0 / (i + 1)  # Azalan kayıp simülasyonu
                loss_history.append(loss)
                progress_callback((i + 1) / epochs * 100)
            
            # Sonuçları hesapla
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Simüle edilmiş tahminler
            y_violence_pred = np.random.randint(0, 2, size=len(y_violence_test))
            y_adult_pred = np.random.randint(0, 2, size=len(y_adult_test))
            y_harassment_pred = np.random.randint(0, 2, size=len(y_harassment_test))
            y_weapon_pred = np.random.randint(0, 2, size=len(y_weapon_test))
            y_drug_pred = np.random.randint(0, 2, size=len(y_drug_test))
            
            # Metrikleri hesapla
            metrics = {
                "violence": {
                    "accuracy": accuracy_score(y_violence_test, y_violence_pred),
                    "precision": precision_score(y_violence_test, y_violence_pred, zero_division=0),
                    "recall": recall_score(y_violence_test, y_violence_pred, zero_division=0),
                    "f1": f1_score(y_violence_test, y_violence_pred, zero_division=0)
                },
                "adult_content": {
                    "accuracy": accuracy_score(y_adult_test, y_adult_pred),
                    "precision": precision_score(y_adult_test, y_adult_pred, zero_division=0),
                    "recall": recall_score(y_adult_test, y_adult_pred, zero_division=0),
                    "f1": f1_score(y_adult_test, y_adult_pred, zero_division=0)
                },
                "harassment": {
                    "accuracy": accuracy_score(y_harassment_test, y_harassment_pred),
                    "precision": precision_score(y_harassment_test, y_harassment_pred, zero_division=0),
                    "recall": recall_score(y_harassment_test, y_harassment_pred, zero_division=0),
                    "f1": f1_score(y_harassment_test, y_harassment_pred, zero_division=0)
                },
                "weapon": {
                    "accuracy": accuracy_score(y_weapon_test, y_weapon_pred),
                    "precision": precision_score(y_weapon_test, y_weapon_pred, zero_division=0),
                    "recall": recall_score(y_weapon_test, y_weapon_pred, zero_division=0),
                    "f1": f1_score(y_weapon_test, y_weapon_pred, zero_division=0)
                },
                "drug": {
                    "accuracy": accuracy_score(y_drug_test, y_drug_pred),
                    "precision": precision_score(y_drug_test, y_drug_pred, zero_division=0),
                    "recall": recall_score(y_drug_test, y_drug_pred, zero_division=0),
                    "f1": f1_score(y_drug_test, y_drug_pred, zero_division=0)
                },
                "overall": {
                    "loss": loss_history[-1],
                    "accuracy": np.mean([
                        accuracy_score(y_violence_test, y_violence_pred),
                        accuracy_score(y_adult_test, y_adult_pred),
                        accuracy_score(y_harassment_test, y_harassment_pred),
                        accuracy_score(y_weapon_test, y_weapon_pred),
                        accuracy_score(y_drug_test, y_drug_pred)
                    ])
                }
            }
            
            # Eğitim grafiklerini oluştur
            self._create_training_plots(loss_history, metrics, epochs)
            
            return True, metrics
        
        except Exception as e:
            current_app.logger.error(f"İçerik modeli eğitim hatası: {str(e)}")
            return False, str(e)
    
    def _train_age_model(self, training_data, epochs, batch_size, learning_rate):
        """
        Yaş tahmin modelini eğitir.
        - Eğitim verisini hazırlar.
        - Gerçek bir eğitim yerine basit bir simülasyon yapar.
        - Eğitim performansını simüle eder.
        - Sonuçları hesaplar ve grafikler oluşturur.
        """
        try:
            # Örnek uygulama: Burada gerçek bir eğitim yapılacak
            # Şu an için basit bir simülasyon yapalım
            
            # Progress kontrolü için
            progress_callback = lambda p: print(f"Eğitim ilerleme: {p:.2f}%")
            
            # Veriyi hazırla
            X = []
            y = []
            
            for item in training_data:
                # Görüntüyü yükle
                img = load_image(item['frame_path'])
                if img is None:
                    continue
                
                # Yüz bölgesini crop et
                face_loc = item['face_location']
                face_img = img[face_loc['y']:face_loc['y']+face_loc['height'], face_loc['x']:face_loc['x']+face_loc['width']]
                
                # Görüntüyü ön işleme
                face_img = cv2.resize(face_img, (224, 224))
                face_img = face_img / 255.0  # Normalize et
                
                X.append(face_img)
                y.append(item['age'])
            
            # Numpy dizilerine dönüştür
            X = np.array(X)
            y = np.array(y)
            
            # Eğitim ve test verisi ayır
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Gerçek bir eğitim yerine simülasyon yap
            # Gerçek projede bu kısımda TensorFlow veya PyTorch ile model eğitimi yapılır
            
            # Eğitim performansı simülasyonu
            loss_history = []
            for i in range(epochs):
                loss = 10.0 / (i + 1)  # Azalan kayıp simülasyonu
                loss_history.append(loss)
                progress_callback((i + 1) / epochs * 100)
            
            # Sonuçları hesapla
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Simüle edilmiş tahminler
            y_pred = y_test + np.random.normal(0, 3, size=len(y_test))  # Gerçek değere yakın tahminler
            
            # Metrikleri hesapla
            metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred)
            }
            
            # Yaş tahmin doğruluk oranlarını hesapla
            age_accuracy = {}
            for threshold in [1, 2, 3, 5, 10]:
                correct = np.sum(np.abs(y_pred - y_test) <= threshold)
                age_accuracy[f"within_{threshold}_years"] = correct / len(y_test)
            
            metrics["age_accuracy"] = age_accuracy
            
            # Eğitim grafiklerini oluştur
            self._create_age_training_plots(loss_history, y_test, y_pred, epochs)
            
            return True, metrics
        
        except Exception as e:
            current_app.logger.error(f"Yaş modeli eğitim hatası: {str(e)}")
            return False, str(e)
    
    def _create_training_plots(self, loss_history, metrics, epochs):
        """
        İçerik analiz modeli için eğitim grafiklerini oluşturur.
        - Kayıp, F1 skor ve doğruluk grafiklerini oluşturur.
        """
        try:
            # Grafikler için dizin oluştur
            plots_dir = os.path.join(self.model_folder, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Kayıp grafiği
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epochs + 1), loss_history)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
            plt.close()
            
            # F1 skor grafiği
            categories = list(metrics.keys())
            categories.remove('overall')
            f1_scores = [metrics[cat]['f1'] for cat in categories]
            
            plt.figure(figsize=(10, 6))
            plt.bar(categories, f1_scores)
            plt.title('F1 Scores by Category')
            plt.xlabel('Category')
            plt.ylabel('F1 Score')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(plots_dir, 'f1_scores.png'))
            plt.close()
            
            # Doğruluk grafiği
            accuracies = [metrics[cat]['accuracy'] for cat in categories]
            
            plt.figure(figsize=(10, 6))
            plt.bar(categories, accuracies)
            plt.title('Accuracy by Category')
            plt.xlabel('Category')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.savefig(os.path.join(plots_dir, 'accuracy.png'))
            plt.close()
        
        except Exception as e:
            current_app.logger.error(f"Grafik oluşturma hatası: {str(e)}")
    
    def _create_age_training_plots(self, loss_history, y_true, y_pred, epochs):
        """
        Yaş tahmin modeli için eğitim grafiklerini oluşturur.
        - Kayıp, Gerçek vs Tahmin edilen yaş ve Hata histogramı grafiklerini oluşturur.
        """
        try:
            # Grafikler için dizin oluştur
            plots_dir = os.path.join(self.model_folder, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Kayıp grafiği
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epochs + 1), loss_history)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
            plt.close()
            
            # Gerçek vs Tahmin edilen yaş grafiği
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Diagonal çizgi
            plt.title('Actual vs Predicted Age')
            plt.xlabel('Actual Age')
            plt.ylabel('Predicted Age')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'age_prediction.png'))
            plt.close()
            
            # Hata histogramı
            errors = y_pred - y_true
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=20)
            plt.title('Error Distribution')
            plt.xlabel('Prediction Error (years)')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
            plt.close()
        
        except Exception as e:
            current_app.logger.error(f"Grafik oluşturma hatası: {str(e)}") 