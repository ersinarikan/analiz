import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open_clip
import numpy as np
from PIL import Image
import json
import datetime
import logging
from flask import current_app
from app import db
from app.models.content import ModelVersion
from app.models.feedback import Feedback
from app.models.analysis import Analysis
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

logger = logging.getLogger(__name__)

class ContentTrainingService:
    """OpenCLIP modeli için fine-tuning servisi"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
        
    def load_base_model(self):
        """Base OpenCLIP modelini yükle"""
        try:
            active_clip_model_path = current_app.config['OPENCLIP_MODEL_ACTIVE_PATH']
            clip_model_name = current_app.config['OPENCLIP_MODEL_NAME'].split('_')[0]
            pretrained_weights_path = os.path.join(active_clip_model_path, 'open_clip_pytorch_model.bin')
            
            if not os.path.exists(pretrained_weights_path):
                # Fallback to base model
                base_clip_model_path = current_app.config['OPENCLIP_MODEL_BASE_PATH']
                pretrained_weights_path = os.path.join(base_clip_model_path, 'open_clip_pytorch_model.bin')
                
            logger.info(f"Loading OpenCLIP model for training: {clip_model_name}")
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=clip_model_name,
                pretrained=pretrained_weights_path,
                device=self.device,
                jit=False
            )
            
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(clip_model_name)
            
            logger.info("Base OpenCLIP model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            return False
    
    def prepare_training_data(self, min_samples=50):
        """Eğitim verisini hazırla"""
        try:
            # Content feedback'leri al
            feedbacks = db.session.query(Feedback).filter(
                Feedback.feedback_type.in_(['violence', 'adult_content', 'harassment', 'weapon', 'drug']),
                Feedback.feedback_source == 'MANUAL_USER'
            ).all()
            
            if len(feedbacks) < min_samples:
                logger.warning(f"Insufficient feedback data: {len(feedbacks)} < {min_samples}")
                return None
            
            # Veriyi kategorilere göre organize et
            training_data = []
            category_counts = {cat: 0 for cat in self.categories}
            
            for feedback in feedbacks:
                if not feedback.frame_path or not os.path.exists(feedback.frame_path):
                    continue
                
                # Binary label oluştur
                label = 1 if feedback.get_category_feedback() == 'positive' else 0
                category = feedback.feedback_type
                
                if category in self.categories:
                    training_data.append({
                        'image_path': feedback.frame_path,
                        'category': category,
                        'label': label,
                        'feedback_id': feedback.id
                    })
                    category_counts[category] += 1
            
            logger.info(f"Prepared {len(training_data)} training samples")
            logger.info(f"Category distribution: {category_counts}")
            
            return {
                'data': training_data,
                'category_counts': category_counts,
                'total_samples': len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None
    
    def create_classification_head(self, num_categories=5):
        """CLIP feature'ları için classification head oluştur"""
        feature_dim = self.model.visual.output_dim  # CLIP visual encoder output dimension
        
        return nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_categories),
            nn.Sigmoid()  # Multi-label classification için
        ).to(self.device)
    
    def train_model(self, training_data, params):
        """Modeli eğit"""
        try:
            if not self.load_base_model():
                raise Exception("Base model loading failed")
            
            # Classification head oluştur
            classifier = self.create_classification_head()
            
            # Dataset ve DataLoader oluştur
            dataset = ContentTrainingDataset(
                training_data['data'], 
                self.preprocess, 
                self.categories
            )
            
            # Train/validation split
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size', 16), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params.get('batch_size', 16), shuffle=False)
            
            # Loss function ve optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(classifier.parameters(), lr=params.get('learning_rate', 0.001))
            
            # CLIP modelini freeze et (sadece classification head'i eğit)
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Training loop
            num_epochs = params.get('epochs', 10)
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            logger.info(f"Starting training: {num_epochs} epochs, {len(train_loader)} batches")
            
            for epoch in range(num_epochs):
                # Training phase
                classifier.train()
                train_loss = 0.0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # CLIP feature extraction (frozen)
                    with torch.no_grad():
                        features = self.model.encode_image(images)
                        features = features / features.norm(dim=-1, keepdim=True)
                    
                    # Classification
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                avg_train_loss = train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                classifier.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        features = self.model.encode_image(images)
                        features = features / features.norm(dim=-1, keepdim=True)
                        
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        all_preds.extend(outputs.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Calculate metrics
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                
                # Binary predictions (threshold = 0.5)
                binary_preds = (all_preds > 0.5).astype(int)
                
                accuracy = accuracy_score(all_labels.flatten(), binary_preds.flatten())
                precision = precision_score(all_labels.flatten(), binary_preds.flatten(), average='weighted', zero_division=0)
                recall = recall_score(all_labels.flatten(), binary_preds.flatten(), average='weighted', zero_division=0)
                f1 = f1_score(all_labels.flatten(), binary_preds.flatten(), average='weighted', zero_division=0)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(classifier.state_dict(), 'best_classifier.pth')
            
            # Load best model
            classifier.load_state_dict(torch.load('best_classifier.pth'))
            
            # Final metrics
            final_metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'best_val_loss': float(best_val_loss)
            }
            
            return {
                'model': classifier,
                'metrics': final_metrics,
                'training_samples': len(train_dataset),
                'validation_samples': len(val_dataset),
                'history': {
                    'train_loss': train_losses,
                    'val_loss': val_losses
                },
                'used_feedback_ids': [item['feedback_id'] for item in training_data['data']]
            }
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
    
    def save_model_version(self, classifier, training_result):
        """Eğitilmiş modeli versiyonla birlikte kaydet"""
        try:
            # Yeni versiyon numarası
            last_version = db.session.query(ModelVersion).filter_by(
                model_type='content'
            ).order_by(ModelVersion.version.desc()).first()
            
            new_version = 1 if not last_version else last_version.version + 1
            
            # Versiyon klasörü oluştur
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            version_name = f"v{new_version}_{timestamp}"
            
            version_path = os.path.join(
                current_app.config['OPENCLIP_MODEL_VERSIONS_PATH'],
                version_name
            )
            os.makedirs(version_path, exist_ok=True)
            
            # Base CLIP modelini kopyala
            base_model_path = os.path.join(
                current_app.config['OPENCLIP_MODEL_BASE_PATH'],
                'open_clip_pytorch_model.bin'
            )
            version_model_path = os.path.join(version_path, 'open_clip_pytorch_model.bin')
            shutil.copy2(base_model_path, version_model_path)
            
            # Classification head'i kaydet
            classifier_path = os.path.join(version_path, 'classification_head.pth')
            torch.save(classifier.state_dict(), classifier_path)
            
            # Metadata kaydet
            metadata = {
                'version': new_version,
                'version_name': version_name,
                'created_at': timestamp,
                'model_type': 'content',
                'base_model': 'OpenCLIP-ViT-H-14-378-quickgelu',
                'training_metrics': training_result['metrics'],
                'training_samples': training_result['training_samples'],
                'validation_samples': training_result['validation_samples']
            }
            
            with open(os.path.join(version_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Veritabanına kaydet
            model_version = ModelVersion(
                model_type='content',
                version=new_version,
                version_name=version_name,
                created_at=datetime.datetime.now(),
                metrics=training_result['metrics'],
                is_active=False,  # Manuel olarak aktifleştirilmeli
                training_samples=training_result['training_samples'],
                validation_samples=training_result['validation_samples'],
                epochs=len(training_result['history']['train_loss']),
                file_path=version_path,
                weights_path=classifier_path,
                used_feedback_ids=training_result['used_feedback_ids']
            )
            
            db.session.add(model_version)
            db.session.commit()
            
            logger.info(f"Model version {version_name} saved successfully")
            return model_version
            
        except Exception as e:
            logger.error(f"Error saving model version: {str(e)}")
            db.session.rollback()
            raise


class ContentTrainingDataset(Dataset):
    """OpenCLIP content training için dataset sınıfı"""
    
    def __init__(self, data, preprocess, categories):
        self.data = data
        self.preprocess = preprocess
        self.categories = categories
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and preprocess image
        image = Image.open(item['image_path']).convert('RGB')
        image = self.preprocess(image)
        
        # Create multi-label vector
        labels = torch.zeros(len(self.categories))
        if item['category'] in self.categories:
            cat_idx = self.categories.index(item['category'])
            labels[cat_idx] = float(item['label'])
        
        return image, labels 