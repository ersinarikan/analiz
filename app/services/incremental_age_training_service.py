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

logger = logging.getLogger('app.incremental_age_training')

class IncrementalAgeTrainingService:
    """
    Incremental Learning için Age Model Training Service
    
    Architecture:
    - Base Model: UTKFace ile eğitilmiş frozen model
    - Fine-tuning Layer: Sadece feedback verisiyle eğitilen katman
    - Transfer Learning: Base model weights korunur
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        logger.info(f"IncrementalAgeTrainingService initialized with device: {self.device}")
    
    def load_base_model(self):
        """
        Base model'i yükle (UTKFace ile eğitilmiş)
        
        Returns:
            CustomAgeHead: Yüklenmiş base model
        """
        logger.info("Loading base age model...")
        
        # Base model path
        base_model_dir = os.path.join(
            current_app.config['MODELS_FOLDER'],
            'age',
            'custom_age_head',
            'base_model'
        )
        
        if not os.path.exists(base_model_dir):
            logger.error(f"Base model directory not found: {base_model_dir}")
            raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
        
        # Model dosyasını bul
        pth_files = [f for f in os.listdir(base_model_dir) if f.endswith('.pth')]
        if not pth_files:
            logger.error(f"No .pth file found in base model directory: {base_model_dir}")
            raise FileNotFoundError(f"No base model file found in: {base_model_dir}")
        
        model_path = os.path.join(base_model_dir, pth_files[0])
        logger.info(f"Loading base model from: {model_path}")
        
        try:
            # Model checkpoint'ini yükle
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # Model konfigürasyonunu al
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                base_model = CustomAgeHead(
                    input_dim=model_config['input_dim'],
                    hidden_dims=model_config['hidden_dims'],
                    output_dim=model_config['output_dim']
                )
            else:
                # Varsayılan konfigürasyon
                base_model = CustomAgeHead(input_dim=512, hidden_dims=[256, 128], output_dim=1)
            
            # Model ağırlıklarını yükle
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                base_model.load_state_dict(checkpoint)
            
            base_model.to(self.device)
            base_model.eval()
            
            logger.info(f"✅ Base model successfully loaded from: {model_path}")
            return base_model
            
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def create_incremental_model(self, base_model, freeze_base=True):
        """
        Incremental learning için model oluştur
        
        Args:
            base_model: Frozen base model
            freeze_base: Base model katmanlarını dondur
            
        Returns:
            nn.Module: Incremental learning modeli
        """
        logger.info("Creating incremental learning model...")
        
        class IncrementalAgeModel(nn.Module):
            def __init__(self, base_model, freeze_base=True):
                super().__init__()
                self.base_model = base_model
                
                # Base model'i dondir
                if freeze_base:
                    for param in self.base_model.parameters():
                        param.requires_grad = False
                    logger.info("Base model frozen for incremental training")
                
                # Fine-tuning layer - embedding seviyesinde 
                # Base model embedding'i al, fine-tuning yap
                self.fine_tune_layer = nn.Sequential(
                    nn.Linear(512, 128),  # Embedding -> hidden
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)     # Final output
                )
                
                # Mixing weight - base vs fine-tune
                self.mix_weight = nn.Parameter(torch.tensor(0.8))  # Başlangıçta base model ağırlıklı
                
            def forward(self, x):
                # Base model prediction
                base_pred = self.base_model(x)
                
                # Fine-tuning prediction directly from embedding
                fine_tune_pred = self.fine_tune_layer(x)
                
                # Weighted mixing: base_weight * base + (1-base_weight) * fine_tune
                # mix_weight=0.8 means 80% base, 20% fine-tune initially
                final_pred = self.mix_weight * base_pred + (1 - self.mix_weight) * fine_tune_pred
                
                return final_pred
        
        incremental_model = IncrementalAgeModel(base_model, freeze_base)
        incremental_model.to(self.device)
        
        logger.info("✅ Incremental model created with fine-tuning layers")
        return incremental_model
    
    def prepare_feedback_data(self, min_samples=2):
        """
        Sadece feedback verilerini hazırla (UTKFace değil!)
        
        Args:
            min_samples: Minimum feedback sayısı
            
        Returns:
            dict: Feedback training data
        """
        logger.info("Preparing feedback-only training data...")
        
        # Sadece yeni feedback'leri al
        feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).filter(
            Feedback.embedding.isnot(None)
        ).filter(
            # Daha önce eğitimde kullanılmamış
            db.or_(
                Feedback.training_status.is_(None),
                Feedback.training_status != 'used_in_training'
            )
        ).all()
        
        logger.info(f"Found {len(feedbacks)} new feedback records")
        
        if len(feedbacks) < min_samples:
            logger.warning(f"Insufficient feedback data: {len(feedbacks)} < {min_samples}")
            return None
        
        # Person ID bazında deduplicate
        person_feedbacks = {}
        
        for feedback in feedbacks:
            person_id = feedback.person_id
            if not person_id:
                continue
                
            if person_id not in person_feedbacks:
                person_feedbacks[person_id] = feedback
            elif (feedback.feedback_source == 'MANUAL_USER' and 
                  person_feedbacks[person_id].feedback_source != 'MANUAL_USER'):
                person_feedbacks[person_id] = feedback
            elif (feedback.feedback_source == 'MANUAL_USER' and 
                  person_feedbacks[person_id].feedback_source == 'MANUAL_USER'):
                if feedback.created_at > person_feedbacks[person_id].created_at:
                    person_feedbacks[person_id] = feedback
        
        # Veriyi hazırla
        embeddings = []
        ages = []
        sources = []
        confidence_scores = []
        feedback_ids = []
        
        for person_id, feedback in person_feedbacks.items():
            try:
                # Embedding
                if isinstance(feedback.embedding, str):
                    embedding = np.array([float(x) for x in feedback.embedding.split(',')])
                else:
                    continue
                
                # Age
                if feedback.feedback_source == 'MANUAL_USER':
                    age = feedback.corrected_age
                    source = 'manual'
                    confidence = 1.0
                else:
                    age = feedback.pseudo_label_original_age
                    source = 'pseudo'
                    confidence = feedback.pseudo_label_clip_confidence or 1.0
                
                if age is None or age < 0 or age > 100:
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
            logger.warning(f"Insufficient valid feedback data: {len(embeddings)} < {min_samples}")
            return None
        
        manual_count = sources.count('manual')
        pseudo_count = sources.count('pseudo')
        
        logger.info(f"✅ Prepared {len(embeddings)} feedback samples: {manual_count} manual, {pseudo_count} pseudo")
        logger.info(f"Age range: {min(ages):.1f} - {max(ages):.1f}, mean: {np.mean(ages):.1f}")
        
        return {
            'embeddings': np.array(embeddings),
            'ages': np.array(ages),
            'sources': sources,
            'confidence_scores': np.array(confidence_scores),
            'feedback_ids': feedback_ids
        }
    
    def train_incremental_model(self, feedback_data, params=None):
        """
        Incremental training yap - sadece feedback verileriyle
        
        Args:
            feedback_data: prepare_feedback_data() sonucu
            params: Training parametreleri
            
        Returns:
            dict: Training sonuçları
        """
        if params is None:
            params = {
                'epochs': 10,          # Çok daha az epoch
                'batch_size': 8,       # Küçük batch size
                'learning_rate': 0.0001,  # Düşük learning rate
                'test_size': 0.2,
                'patience': 5
            }
        
        logger.info(f"Starting incremental training with params: {params}")
        
        # Base model'i yükle
        base_model = self.load_base_model()
        
        # Incremental model oluştur
        incremental_model = self.create_incremental_model(base_model)
        
        # Veriyi hazırla
        X = feedback_data['embeddings']
        y = feedback_data['ages'].reshape(-1, 1)
        confidence_weights = feedback_data['confidence_scores']
        
        # Train/validation split
        if len(X) > 4:  # En az 4 sample varsa split yap
            indices = np.arange(len(X))
            train_idx, val_idx = train_test_split(indices, test_size=params['test_size'], random_state=42)
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            weights_train = confidence_weights[train_idx]
        else:
            # Çok az veri varsa hepsini training'de kullan
            X_train, X_val = X, X
            y_train, y_val = y, y
            weights_train = confidence_weights
        
        # Normalization (critical!)
        X_train_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        X_val_norm = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
        
        # PyTorch tensors
        X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        weights_tensor = torch.tensor(weights_train, dtype=torch.float32)
        
        X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        
        # Optimizer - sadece fine-tuning parametreleri
        optimizer = optim.Adam(
            [p for p in incremental_model.parameters() if p.requires_grad], 
            lr=params['learning_rate']
        )
        
        criterion = nn.MSELoss(reduction='none')  # Weighted loss için
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting incremental training loop...")
        
        for epoch in range(params['epochs']):
            # Training phase
            incremental_model.train()
            train_losses = []
            
            for batch_X, batch_y, batch_weights in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_weights = batch_weights.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = incremental_model(batch_X)
                loss_per_sample = criterion(outputs, batch_y).squeeze()
                
                # Weighted loss
                weighted_loss = (loss_per_sample * batch_weights).mean()
                
                weighted_loss.backward()
                optimizer.step()
                
                train_losses.append(weighted_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation phase
            incremental_model.eval()
            with torch.no_grad():
                X_val_device = X_val_tensor.to(self.device)
                y_val_device = y_val_tensor.to(self.device)
                
                val_outputs = incremental_model(X_val_device)
                val_loss = criterion(val_outputs, y_val_device).mean().item()
                val_mae = torch.abs(val_outputs - y_val_device).mean().item()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            
            logger.info(f"Epoch {epoch+1}/{params['epochs']}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val MAE: {val_mae:.2f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = incremental_model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= params['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        incremental_model.load_state_dict(best_model_state)
        
        # Final evaluation
        incremental_model.eval()
        with torch.no_grad():
            X_val_device = X_val_tensor.to(self.device)
            y_val_device = y_val_tensor.to(self.device)
            
            val_outputs = incremental_model(X_val_device)
            final_mae = torch.abs(val_outputs - y_val_device).mean().item()
            final_mse = ((val_outputs - y_val_device) ** 2).mean().item()
            
            # Accuracy within tolerance
            tolerance = 3.0  # years
            within_tolerance = (torch.abs(val_outputs - y_val_device) <= tolerance).float().mean().item()
        
        metrics = {
            'mae': final_mae,
            'mse': final_mse,
            'accuracy_3years': within_tolerance * 100,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
        
        logger.info(f"✅ Incremental training completed!")
        logger.info(f"Final metrics: MAE={final_mae:.3f}, 3-year accuracy={within_tolerance*100:.1f}%")
        
        return {
            'model': incremental_model,
            'metrics': metrics,
            'history': history,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'used_feedback_ids': feedback_data['feedback_ids']
        }
    
    def save_incremental_model(self, model, training_result, version_name=None):
        """
        Incremental model'i kaydet
        
        Args:
            model: Eğitilmiş incremental model
            training_result: Training sonucu
            version_name: Version adı
            
        Returns:
            ModelVersion: Kaydedilen model versiyonu
        """
        logger.info("Saving incremental model...")
        
        # Version name
        if version_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            feedback_count = len(training_result['used_feedback_ids'])
            version_name = f"incremental_v1_{feedback_count}feedback_{timestamp}"
        
        # Save directory
        version_dir = os.path.join(
            current_app.config['MODELS_FOLDER'],
            'age',
            'custom_age_head',
            'versions',
            version_name
        )
        os.makedirs(version_dir, exist_ok=True)
        
        # Model path
        model_path = os.path.join(version_dir, 'model.pth')
        
                 # Extract fine-tuning part for saving
         # We only save the incremental part, not the base model
         fine_tune_state = {
             'fine_tune_layer': model.fine_tune_layer.state_dict(),
             'mix_weight': model.mix_weight.data,
             'model_type': 'incremental',
             'base_model_reference': 'base_model'  # Reference to base model
         }
        
        torch.save({
            'model_state_dict': model.state_dict(),  # Full model for loading
            'fine_tune_only': fine_tune_state,       # Just fine-tuning part
            'model_config': {
                'model_type': 'incremental',
                'base_model_path': 'base_model',
                'fine_tune_layers': [32, 16, 1]
            }
        }, model_path)
        
        # Training details
        details_path = os.path.join(version_dir, 'training_details.json')
        with open(details_path, 'w') as f:
            json.dump({
                'metrics': training_result['metrics'],
                'history': training_result['history'],
                'training_samples': training_result['training_samples'],
                'validation_samples': training_result['validation_samples'],
                'training_type': 'incremental_feedback',
                'training_date': datetime.now().isoformat(),
                'feedback_count': len(training_result['used_feedback_ids'])
            }, f, indent=4)
        
        # Database entry
        last_version = ModelVersion.query.filter_by(
            model_type='age'
        ).order_by(ModelVersion.version.desc()).first()
        
        new_version_num = 1 if last_version is None else last_version.version + 1
        
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
            is_active=False,
            created_at=datetime.now(),
            used_feedback_ids=training_result['used_feedback_ids']
        )
        
        db.session.add(model_version)
        db.session.commit()
        
        logger.info(f"✅ Incremental model saved: {version_name}")
        return model_version
    
    def run_incremental_training(self, min_feedback_samples=2):
        """
        Full incremental training pipeline
        
        Args:
            min_feedback_samples: Minimum feedback count to start training
            
        Returns:
            dict: Training results or None if insufficient data
        """
        logger.info("🚀 Starting incremental age training pipeline...")
        
        # 1. Prepare feedback data
        feedback_data = self.prepare_feedback_data(min_feedback_samples)
        if feedback_data is None:
            logger.warning("Insufficient feedback data for incremental training")
            return None
        
        # 2. Train incremental model
        training_result = self.train_incremental_model(feedback_data)
        
        # 3. Save model
        model_version = self.save_incremental_model(
            training_result['model'], 
            training_result
        )
        
        # 4. Mark feedback as used
        from app.services.age_training_service import AgeTrainingService
        age_service = AgeTrainingService()
        age_service.mark_training_data_used(
            training_result['used_feedback_ids'],
            model_version.version_name
        )
        
        logger.info(f"✅ Incremental training pipeline completed!")
        logger.info(f"Model version: {model_version.version_name}")
        logger.info(f"Final MAE: {training_result['metrics']['mae']:.3f}")
        
        return {
            'model_version': model_version,
            'training_result': training_result,
            'feedback_data': feedback_data
        } 