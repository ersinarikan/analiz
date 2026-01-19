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
from config import Config
from app.utils.model_utils import load_torch_model

logger = logging.getLogger('app.incremental_age_training_v2')

class IncrementalAgeTrainingServiceV2:
    """
    Artımsal yaş tahmini modelinin ikinci versiyonu için eğitim ve güncelleme servis sınıfı.
    - Yeni verilerle modelin güncellenmesini ve performans takibini sağlar.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        logger.info(f"IncrementalAgeTrainingServiceV2 initialized with device: {self.device}")
    
    def load_base_model(self) -> nn.Module:
        """Base model'i yükle (UTKFace ile eğitilmiş)"""
        logger.info("Loading base age model...")
        base_model_dir = os.path.join(
            current_app.config['MODELS_FOLDER'],
            'age', 'custom_age_head', 'base_model'
        )
        if not os.path.exists(base_model_dir):
            raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
        pth_files = [f for f in os.listdir(base_model_dir) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No base model file found in: {base_model_dir}")
        model_path = os.path.join(base_model_dir, pth_files[0])
        config_keys = ['input_dim', 'hidden_dims', 'output_dim']
        default_config = {'input_dim': 512, 'hidden_dims': [256, 128], 'output_dim': 1}
        return load_torch_model(model_path, CustomAgeHead, config_keys, self.device, default_config)
    
    def create_incremental_model(self, base_model: nn.Module) -> nn.Module:
        """Improved incremental learning model"""
        logger.info("Creating improved incremental learning model...")
        
        class ImprovedIncrementalAgeModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
                # Freeze base model
                for param in self.base_model.parameters():
                    param.requires_grad = False
                
                # Fine-tuning branch - works directly on embeddings
                self.fine_tune_branch = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
                # Learnable mixing weight (0=all base, 1=all fine-tune)
                self.mix_weight = nn.Parameter(torch.tensor(0.2))  # Start with 20% fine-tune
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Base model prediction (frozen)
                with torch.no_grad():
                    base_pred = self.base_model(x)
                
                # Fine-tuning branch prediction
                fine_pred = self.fine_tune_branch(x)
                
                # Learnable weighted combination
                # Sigmoid ensures weight is between 0 and 1
                weight = torch.sigmoid(self.mix_weight)
                final_pred = (1 - weight) * base_pred + weight * fine_pred
                
                return final_pred
        
        model = ImprovedIncrementalAgeModel(base_model)
        model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("✅ Improved incremental model created")
        logger.info(f"   Total params: {total_params:,}")
        logger.info(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        return model
    
    def prepare_feedback_data(self, min_samples: int = 2) -> dict | None:
        """Feedback verilerini hazırla"""
        logger.info("Preparing feedback data...")
        
        feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).filter(
            Feedback.embedding.isnot(None)
        ).filter(
            db.or_(
                Feedback.training_status.is_(None),
                Feedback.training_status != 'used_in_training'
            )
        ).all()
        
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
        
        # Data preparation
        embeddings: list[np.ndarray] = []
        ages: list[float] = []
        sources: list[str] = []
        confidence_scores: list[float] = []
        feedback_ids: list[int] = []
        
        for person_id, feedback in person_feedbacks.items():
            try:
                if isinstance(feedback.embedding, str):
                    embedding = np.array([float(x) for x in feedback.embedding.split(',')])
                else:
                    continue
                
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
        
        manual_count = sources.count('manual')
        pseudo_count = sources.count('pseudo')
        
        logger.info(f"✅ Prepared {len(embeddings)} samples: {manual_count} manual, {pseudo_count} pseudo")
        
        return {
            'embeddings': np.array(embeddings),
            'ages': np.array(ages),
            'sources': sources,
            'confidence_scores': np.array(confidence_scores),
            'feedback_ids': feedback_ids
        }
    
    def train_incremental_model(self, feedback_data: dict) -> dict:
        """Improved incremental training"""
        params = Config.DEFAULT_TRAINING_PARAMS.copy()
        
        logger.info(f"Starting improved incremental training: {params}")
        
        # Load models
        base_model = self.load_base_model()
        model = self.create_incremental_model(base_model)
        
        # Prepare data
        X = feedback_data['embeddings']
        y = feedback_data['ages'].reshape(-1, 1)
        weights = feedback_data['confidence_scores']
        
        # Split
        if len(X) > 4:
            indices = np.arange(len(X))
            train_idx, val_idx = train_test_split(indices, test_size=params['test_size'], random_state=42)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            weights_train = weights[train_idx]
        else:
            X_train, X_val = X, X
            y_train, y_val = y, y
            weights_train = weights
        
        # Normalize
        X_train_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        X_val_norm = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
        
        # Tensors
        X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        weights_tensor = torch.tensor(weights_train, dtype=torch.float32).to(self.device)
        
        X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=params['learning_rate']
        )
        criterion = nn.MSELoss(reduction='none')
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            model.train()
            
            # Training
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss_per_sample = criterion(outputs, y_train_tensor).squeeze()
            weighted_loss = (loss_per_sample * weights_tensor).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).mean().item()
                val_mae = torch.abs(val_outputs - y_val_tensor).mean().item()
            
            history['train_loss'].append(weighted_loss.item())
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            
            logger.info(f"Epoch {epoch+1:2d}: Train={weighted_loss.item():.4f}, "
                       f"Val={val_loss:.4f}, MAE={val_mae:.2f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= params['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            final_mae = torch.abs(val_outputs - y_val_tensor).mean().item()
            final_mse = ((val_outputs - y_val_tensor) ** 2).mean().item()
            tolerance_3y = (torch.abs(val_outputs - y_val_tensor) <= 3.0).float().mean().item()
        
        metrics = {
            'mae': final_mae,
            'mse': final_mse,
            'accuracy_3years': tolerance_3y * 100,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
        
        logger.info(f"✅ Training completed! MAE={final_mae:.3f}, 3y-acc={tolerance_3y*100:.1f}%")
        
        return {
            'model': model,
            'metrics': metrics,
            'history': history,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'used_feedback_ids': feedback_data['feedback_ids']
        }
    
    def run_incremental_training(self, min_feedback_samples: int = 2) -> dict | None:
        """Full pipeline"""
        logger.info("🚀 Starting improved incremental training pipeline...")
        
        # Prepare data
        feedback_data = self.prepare_feedback_data(min_feedback_samples)
        if feedback_data is None:
            return None
        
        # Train
        training_result = self.train_incremental_model(feedback_data)
        
        # Simple save (without full versioning for testing)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        feedback_count = len(training_result['used_feedback_ids'])
        version_name = f"incremental_v2_{feedback_count}feedback_{timestamp}"
        
        logger.info(f"✅ Pipeline completed: {version_name}")
        logger.info(f"Final MAE: {training_result['metrics']['mae']:.3f}")
        
        return {
            'version_name': version_name,
            'training_result': training_result,
            'feedback_data': feedback_data
        } 