import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
from flask import current_app
from app import db
from app.models.feedback import Feedback
from app.models.content import ModelVersion
from app.ai.insightface_age_estimator import CustomAgeHead

logger = logging.getLogger(__name__)

class AgeTrainingService:
    """Custom Age modelini geri bildirimlerle eğiten servis"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        logger.info(f"AgeTrainingService initialized with device: {self.device}")
    
    def prepare_training_data(self, min_samples=10):
        """
        Feedback tablosundan eğitim verilerini hazırlar
        
        Returns:
            dict: {
                'embeddings': np.array,
                'ages': np.array,
                'sources': list,  # Her örneğin kaynağı (manual/pseudo)
                'confidence_scores': np.array,  # Pseudo label'lar için güven skorları
                'feedback_ids': list  # Kullanılan feedback ID'leri
            }
        """
        logger.info("Preparing training data from feedback table...")
        
        # Yaş geri bildirimi olan tüm kayıtları al
        feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).filter(
            Feedback.embedding.isnot(None)
        ).all()
        
        logger.info(f"Found {len(feedbacks)} age feedback records with embeddings")
        
        if len(feedbacks) < min_samples:
            logger.warning(f"Insufficient feedback data: {len(feedbacks)} < {min_samples}")
            return None
        
        # Person ID bazında verileri organize et (çakışmaları önlemek için)
        person_feedbacks = {}
        
        for feedback in feedbacks:
            person_id = feedback.person_id
            if not person_id:
                logger.debug(f"Feedback {feedback.id} has no person_id, skipping")
                continue
                
            # Eğer bu person_id için daha önce veri yoksa veya 
            # mevcut veri pseudo iken yeni veri manual ise güncelle
            if person_id not in person_feedbacks:
                logger.debug(f"Person {person_id}: First feedback (ID: {feedback.id}, Source: {feedback.feedback_source})")
                person_feedbacks[person_id] = feedback
            elif (feedback.feedback_source == 'MANUAL_USER' and 
                  person_feedbacks[person_id].feedback_source != 'MANUAL_USER'):
                # Manuel geri bildirim her zaman önceliklidir
                logger.info(f"Person {person_id}: Manual feedback (ID: {feedback.id}) overrides pseudo-label (ID: {person_feedbacks[person_id].id})")
                person_feedbacks[person_id] = feedback
            elif (feedback.feedback_source == 'MANUAL_USER' and 
                  person_feedbacks[person_id].feedback_source == 'MANUAL_USER'):
                # İki manuel geri bildirim varsa, en son olanı kullan
                if feedback.created_at > person_feedbacks[person_id].created_at:
                    logger.info(f"Person {person_id}: Using newer manual feedback (ID: {feedback.id} > {person_feedbacks[person_id].id})")
                    person_feedbacks[person_id] = feedback
                else:
                    logger.debug(f"Person {person_id}: Keeping older manual feedback (ID: {person_feedbacks[person_id].id})")
        
        # Verileri hazırla
        embeddings = []
        ages = []
        sources = []
        confidence_scores = []
        feedback_ids = []
        
        for person_id, feedback in person_feedbacks.items():
            try:
                # Embedding'i string'den numpy array'e dönüştür
                if isinstance(feedback.embedding, str):
                    embedding = np.array([float(x) for x in feedback.embedding.split(',')])
                else:
                    logger.warning(f"Feedback {feedback.id}: embedding is not string, skipping")
                    continue
                
                # Yaş değerini al
                if feedback.feedback_source == 'MANUAL_USER':
                    age = feedback.corrected_age
                    source = 'manual'
                    confidence = 1.0  # Manuel veriler için tam güven
                else:  # PSEUDO_BUFFALO_HIGH_CONF
                    age = feedback.pseudo_label_original_age
                    source = 'pseudo'
                    # Backend'de zaten filtrelenmiş olduğu için güven skorunu olduğu gibi kullan
                    confidence = feedback.pseudo_label_clip_confidence or 1.0
                
                if age is None or age < 0 or age > 100:
                    logger.warning(f"Feedback {feedback.id}: invalid age {age}, skipping")
                    continue
                
                embeddings.append(embedding)
                ages.append(float(age))
                sources.append(source)
                confidence_scores.append(confidence)
                feedback_ids.append(feedback.id)
                
            except Exception as e:
                logger.error(f"Error processing feedback {feedback.id}: {str(e)}")
                continue
        
        if len(embeddings) < min_samples:
            logger.warning(f"Insufficient valid data after processing: {len(embeddings)} < {min_samples}")
            return None
        
        # Veri istatistikleri
        manual_count = sources.count('manual')
        pseudo_count = sources.count('pseudo')
        logger.info(f"Prepared {len(embeddings)} unique person samples: {manual_count} manual, {pseudo_count} pseudo-labeled")
        logger.info(f"Age range: {min(ages):.1f} - {max(ages):.1f}, mean: {np.mean(ages):.1f}")
        
        return {
            'embeddings': np.array(embeddings),
            'ages': np.array(ages),
            'sources': sources,
            'confidence_scores': np.array(confidence_scores),
            'feedback_ids': feedback_ids
        }
    
    def train_model(self, training_data, params=None):
        """
        Custom Age modelini eğitir
        
        Args:
            training_data: prepare_training_data() fonksiyonundan dönen veri
            params: Eğitim parametreleri
            
        Returns:
            dict: Eğitim sonuçları
        """
        if params is None:
            params = {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'hidden_dims': [256, 128],
                'test_size': 0.2,
                'early_stopping_patience': 10
            }
        
        logger.info(f"Starting training with params: {params}")
        
        # Veriyi hazırla
        X = training_data['embeddings']
        y = training_data['ages'].reshape(-1, 1)
        confidence_weights = training_data['confidence_scores']
        
        # Train/validation split (confidence weight'leri koruyarak)
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(indices, test_size=params['test_size'], random_state=42)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        weights_train = confidence_weights[train_idx]
        
        # Veri istatistikleri
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        
        # PyTorch tensörlere dönüştür
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        weights_tensor = torch.FloatTensor(weights_train).to(self.device)
        
        # DataLoader oluştur
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Model oluştur
        input_dim = X.shape[1]  # 512 for InsightFace embeddings
        model = CustomAgeHead(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            output_dim=1
        ).to(self.device)
        
        # Loss ve optimizer
        criterion = nn.MSELoss(reduction='none')  # Weighted loss için
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Eğitim geçmişi
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Eğitim döngüsü
        for epoch in range(params['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y, batch_weights in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Weighted loss
                losses = criterion(outputs, batch_y)
                weighted_loss = (losses * batch_weights.unsqueeze(1)).mean()
                
                weighted_loss.backward()
                optimizer.step()
                
                train_loss += weighted_loss.item() * batch_X.size(0)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y).mean()
                    val_loss += loss.item() * batch_X.size(0)
                    val_mae += torch.abs(outputs - batch_y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            avg_val_mae = val_mae / len(val_loader.dataset)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(avg_val_mae)
            
            logger.info(f"Epoch [{epoch+1}/{params['epochs']}] "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Val MAE: {avg_val_mae:.2f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= params['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # En iyi modeli yükle
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test seti üzerinde final metrikler
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).cpu().numpy()
        
        val_errors = np.abs(val_preds.flatten() - y_val.flatten())
        
        metrics = {
            'mae': float(np.mean(val_errors)),
            'mse': float(np.mean(val_errors ** 2)),
            'rmse': float(np.sqrt(np.mean(val_errors ** 2))),
            'within_3_years': float(np.mean(val_errors <= 3)),
            'within_5_years': float(np.mean(val_errors <= 5)),
            'within_10_years': float(np.mean(val_errors <= 10))
        }
        
        logger.info(f"Final metrics: {metrics}")
        
        return {
            'model': model,
            'metrics': metrics,
            'history': history,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'used_feedback_ids': training_data['feedback_ids']
        }
    
    def save_model_version(self, model, training_result, version_name=None):
        """
        Eğitilmiş modeli yeni bir versiyon olarak kaydet
        
        Args:
            model: Eğitilmiş PyTorch modeli
            training_result: train_model() fonksiyonundan dönen sonuç
            version_name: Opsiyonel versiyon adı
            
        Returns:
            ModelVersion: Oluşturulan model versiyonu
        """
        # Versiyon numarasını belirle
        last_version = ModelVersion.query.filter_by(
            model_type='age'
        ).order_by(ModelVersion.version.desc()).first()
        
        new_version_num = 1 if last_version is None else last_version.version + 1
        
        if version_name is None:
            version_name = f"v{new_version_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Model dosyalarını kaydet
        version_dir = os.path.join(
            current_app.config['MODELS_FOLDER'],
            'age',
            'custom_age_head',
            'versions',
            version_name
        )
        os.makedirs(version_dir, exist_ok=True)
        
        # Model ağırlıklarını kaydet
        model_path = os.path.join(version_dir, 'model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': model.network[0].in_features,
                'hidden_dims': [layer.out_features for layer in model.network if isinstance(layer, nn.Linear)][:-1],
                'output_dim': 1
            }
        }, model_path)
        
        # Eğitim detaylarını kaydet
        details_path = os.path.join(version_dir, 'training_details.json')
        with open(details_path, 'w') as f:
            json.dump({
                'metrics': training_result['metrics'],
                'history': training_result['history'],
                'training_samples': training_result['training_samples'],
                'validation_samples': training_result['validation_samples'],
                'training_date': datetime.now().isoformat()
            }, f, indent=4)
        
        # Veritabanına kaydet
        model_version = ModelVersion(
            model_type='age',
            version=new_version_num,
            version_name=version_name,
            file_path=version_dir,
            weights_path=model_path,
            metrics=training_result['metrics'],
            training_samples=training_result['training_samples'],
            validation_samples=training_result['validation_samples'],
            epochs=len(training_result['history']['train_loss']),
            is_active=False,  # Manuel olarak aktif edilmeli
            created_at=datetime.now(),
            used_feedback_ids=training_result['used_feedback_ids']
        )
        
        db.session.add(model_version)
        db.session.commit()
        
        logger.info(f"Model version saved: {version_name} (ID: {model_version.id})")
        
        return model_version
    
    def activate_model_version(self, version_id):
        """
        Belirli bir model versiyonunu aktif hale getirir
        
        Args:
            version_id: Aktif edilecek ModelVersion ID'si
            
        Returns:
            bool: Başarılı olup olmadığı
        """
        try:
            # Versiyonu bul
            version = ModelVersion.query.filter_by(
                id=version_id,
                model_type='age'
            ).first()
            
            if not version:
                logger.error(f"Model version not found: {version_id}")
                return False
            
            # Mevcut aktif versiyonu devre dışı bırak
            ModelVersion.query.filter_by(
                model_type='age',
                is_active=True
            ).update({'is_active': False})
            
            # Yeni versiyonu aktif et
            version.is_active = True
            db.session.commit()
            
            # Aktif model sembolik linkini güncelle
            active_dir = os.path.join(
                current_app.config['MODELS_FOLDER'],
                'age',
                'custom_age_head',
                'active_model'
            )
            
            # Eski sembolik linki kaldır
            if os.path.exists(active_dir):
                if os.path.islink(active_dir):
                    os.unlink(active_dir)
                else:
                    import shutil
                    shutil.rmtree(active_dir)
            
            # Yeni sembolik link oluştur
            os.symlink(version.file_path, active_dir)
            
            logger.info(f"Activated model version: {version.version_name} (ID: {version_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating model version: {str(e)}")
            db.session.rollback()
            return False
    
    def get_model_versions(self):
        """Tüm Custom Age model versiyonlarını listeler"""
        versions = ModelVersion.query.filter_by(
            model_type='age'
        ).order_by(ModelVersion.created_at.desc()).all()
        
        return [{
            'id': v.id,
            'version': v.version,
            'version_name': v.version_name,
            'is_active': v.is_active,
            'created_at': v.created_at.isoformat(),
            'metrics': v.metrics,
            'training_samples': v.training_samples,
            'validation_samples': v.validation_samples
        } for v in versions] 