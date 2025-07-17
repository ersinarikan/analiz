import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open_clip
import PIL.Image as Image
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from flask import current_app
from sklearn.model_selection import train_test_split

from app import db
from app.models.feedback import Feedback
from app.models.clip_training import CLIPTrainingSession
from app.models.content import ModelVersion
from app.utils.image_utils import load_image

logger = logging.getLogger('app.clip_training_service')

class ContentDataset(Dataset):
    """CLIP fine-tuning için veri seti"""
    
    def __init__(self, image_paths: List[str], captions: List[str], labels: List[Dict], preprocess):
        self.image_paths = image_paths
        self.captions = captions
        self.labels = labels
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Görüntüyü yükle
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.preprocess(image)
        except Exception as e:
            logger.warning(f"Görüntü yüklenemedi {self.image_paths[idx]}: {e}")
            # Boş görüntü oluştur
            image = torch.zeros(3, 224, 224)
        
        caption = self.captions[idx]
        labels = self.labels[idx]
        
        return image, caption, labels

class ClipTrainingService:
    """
    OpenCLIP modeli için fine-tuning servisi
    - Feedback verilerinden training data hazırlar
    - Contrastive learning ile model eğitir
    - Classification head ekler
    - Model versiyonlarını yönetir
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.classification_head = None
        
        logger.info(f"ClipTrainingService initialized on device: {self.device}")
    
    def load_base_model(self):
        """Base OpenCLIP modelini yükle"""
        try:
            logger.info("Base OpenCLIP modeli yükleniyor...")
            
            # Base model yükle
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14-378-quickgelu',
                pretrained='dfn5b',
                device=self.device
            )
            
            # Tokenizer yükle
            tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
            
            self.model = model
            self.tokenizer = tokenizer
            self.preprocess = preprocess
            
            logger.info("✅ Base OpenCLIP modeli başarıyla yüklendi")
            return True
            
        except Exception as e:
            logger.error(f"Base model yükleme hatası: {e}")
            return False
    
    def prepare_training_data(self, min_samples: int = 10) -> Optional[Dict]:
        """Feedback verilerinden training data hazırla"""
        logger.info(f"CLIP training verisi hazırlanıyor (min: {min_samples} örnek)...")
        
        try:
            # Content feedback'lerini al
            feedbacks = db.session.query(Feedback).filter(
                Feedback.feedback_type == 'content',
                Feedback.frame_path.isnot(None)
            ).all()
            
            if len(feedbacks) < min_samples:
                logger.warning(f"Yetersiz feedback: {len(feedbacks)} < {min_samples}")
                return None
            
            # Training data listeler
            image_paths = []
            positive_captions = []
            negative_captions = []
            labels = []
            
            for feedback in feedbacks:
                # Frame path'i tam yola çevir
                if feedback.frame_path:
                    frame_path = os.path.join(current_app.config['STORAGE_FOLDER'], feedback.frame_path)
                    
                    if os.path.exists(frame_path):
                        image_paths.append(frame_path)
                        
                        # Kullanıcı yorumundan pozitif caption oluştur
                        positive_caption = self._create_positive_caption(feedback)
                        positive_captions.append(positive_caption)
                        
                        # Ters caption oluştur (contrastive learning için)
                        negative_caption = self._create_negative_caption(feedback)
                        negative_captions.append(negative_caption)
                        
                        # Label bilgileri
                        label_info = self._extract_labels(feedback)
                        labels.append(label_info)
            
            if len(image_paths) < min_samples:
                logger.warning(f"Geçerli görüntü sayısı yetersiz: {len(image_paths)} < {min_samples}")
                return None
            
            # Training ve validation'a ayır
            train_indices, val_indices = train_test_split(
                range(len(image_paths)), 
                test_size=0.2, 
                random_state=42
            )
            
            training_data = {
                'train_images': [image_paths[i] for i in train_indices],
                'train_positive_captions': [positive_captions[i] for i in train_indices],
                'train_negative_captions': [negative_captions[i] for i in train_indices],
                'train_labels': [labels[i] for i in train_indices],
                
                'val_images': [image_paths[i] for i in val_indices],
                'val_positive_captions': [positive_captions[i] for i in val_indices],
                'val_negative_captions': [negative_captions[i] for i in val_indices],
                'val_labels': [labels[i] for i in val_indices],
                
                'total_samples': len(image_paths),
                'train_samples': len(train_indices),
                'val_samples': len(val_indices)
            }
            
            logger.info(f"✅ Training data hazırlandı: {training_data['total_samples']} örnek")
            logger.info(f"   Train: {training_data['train_samples']}, Val: {training_data['val_samples']}")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Training data hazırlama hatası: {e}")
            return None
    
    def _create_positive_caption(self, feedback: Feedback) -> str:
        """Feedback'ten pozitif caption oluştur"""
        try:
            # Kullanıcı yorumu varsa onu kullan
            if feedback.comment and feedback.comment.strip():
                return feedback.comment.strip()
            
            # Category feedback'ten caption oluştur
            if feedback.category_feedback:
                category_data = feedback.category_feedback
                if isinstance(category_data, str):
                    category_data = json.loads(category_data)
                
                return self._generate_caption_from_categories(category_data, positive=True)
            
            # Default safe caption
            return "safe appropriate content"
            
        except Exception as e:
            logger.warning(f"Pozitif caption oluşturma hatası: {e}")
            return "appropriate content"
    
    def _create_negative_caption(self, feedback: Feedback) -> str:
        """Contrastive learning için negatif caption oluştur"""
        try:
            if feedback.category_feedback:
                category_data = feedback.category_feedback
                if isinstance(category_data, str):
                    category_data = json.loads(category_data)
                
                return self._generate_caption_from_categories(category_data, positive=False)
            
            # Default negative caption
            return "inappropriate violent adult content"
            
        except Exception as e:
            logger.warning(f"Negatif caption oluşturma hatası: {e}")
            return "inappropriate content"
    
    def _generate_caption_from_categories(self, categories: Dict, positive: bool = True) -> str:
        """Kategori feedback'ten caption oluştur"""
        safe_terms = []
        unsafe_terms = []
        
        category_mappings = {
            'violence': ('peaceful non-violent content', 'violent aggressive content'),
            'adult_content': ('family-friendly appropriate content', 'adult explicit content'),
            'harassment': ('respectful positive interaction', 'harassment bullying content'),
            'weapon': ('safe environment without weapons', 'dangerous weapons present'),
            'drug': ('drug-free healthy environment', 'drug substance abuse content')
        }
        
        for category, level in categories.items():
            if category in category_mappings:
                safe_desc, unsafe_desc = category_mappings[category]
                
                if level == 'low':
                    safe_terms.append(safe_desc)
                elif level == 'high':
                    unsafe_terms.append(unsafe_desc)
        
        if positive:
            # Pozitif için safe terms kullan, yoksa genel safe
            return ', '.join(safe_terms) if safe_terms else "safe appropriate content"
        else:
            # Negatif için unsafe terms kullan, yoksa genel unsafe
            return ', '.join(unsafe_terms) if unsafe_terms else "inappropriate harmful content"
    
    def _extract_labels(self, feedback: Feedback) -> Dict:
        """Feedback'ten label bilgilerini çıkart"""
        labels = {
            'violence': 0,
            'adult_content': 0,
            'harassment': 0,
            'weapon': 0,
            'drug': 0,
            'safe': 1  # Default safe
        }
        
        try:
            if feedback.category_feedback:
                category_data = feedback.category_feedback
                if isinstance(category_data, str):
                    category_data = json.loads(category_data)
                
                for category, level in category_data.items():
                    if category in labels:
                        if level == 'high':
                            labels[category] = 1
                            labels['safe'] = 0  # Unsafe content
                        elif level == 'low':
                            labels[category] = 0
        
        except Exception as e:
            logger.warning(f"Label extraction hatası: {e}")
        
        return labels
    
    def create_classification_head(self, num_classes: int = 6) -> nn.Module:
        """CLIP için classification head oluştur"""
        # CLIP'in text encoder çıkış boyutu (ViT-H-14 için 1024)
        clip_dim = self.model.text.text_projection.out_features
        
        classification_head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Multi-label classification için
        ).to(self.device)
        
        return classification_head
    
    def train_model(self, training_data: Dict, training_params: Dict) -> Dict:
        """CLIP modelini fine-tune et"""
        logger.info("CLIP fine-tuning başlıyor...")
        
        try:
            # Base model yükle
            if not self.load_base_model():
                raise Exception("Base model yüklenemedi")
            
            # Classification head oluştur
            self.classification_head = self.create_classification_head()
            
            # Data loaders oluştur
            train_dataset = ContentDataset(
                training_data['train_images'],
                training_data['train_positive_captions'],
                training_data['train_labels'],
                self.preprocess
            )
            
            val_dataset = ContentDataset(
                training_data['val_images'],
                training_data['val_positive_captions'],
                training_data['val_labels'],
                self.preprocess
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_params.get('batch_size', 16),
                shuffle=True,
                num_workers=0  # Windows için 0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_params.get('batch_size', 16),
                shuffle=False,
                num_workers=0
            )
            
            # Optimizer - sadece classification head'i eğit (CLIP frozen)
            optimizer = optim.Adam(
                self.classification_head.parameters(),
                lr=training_params.get('learning_rate', 1e-4)
            )
            
            # Loss function
            criterion = nn.BCELoss()
            
            # Training loop
            history = self._training_loop(
                train_loader, val_loader, optimizer, criterion, training_params
            )
            
            # Model kaydet
            model_path = self._save_trained_model(training_params)
            
            # Training session kaydet
            training_session = self._save_training_session(training_data, training_params, history, model_path)
            
            result = {
                'success': True,
                'training_session_id': training_session.id,
                'model_path': model_path,
                'history': history,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'epochs_completed': len(history['train_loss'])
            }
            
            logger.info("✅ CLIP fine-tuning tamamlandı")
            return result
            
        except Exception as e:
            logger.error(f"CLIP training hatası: {e}")
            return {'success': False, 'error': str(e)}
    
    def _training_loop(self, train_loader, val_loader, optimizer, criterion, params) -> Dict:
        """Training loop implementation"""
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        epochs = params.get('epochs', 10)
        
        # CLIP'i frozen tut
        for param in self.model.parameters():
            param.requires_grad = False
        
        best_val_loss = float('inf')
        patience = params.get('patience', 3)
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.classification_head.train()
            train_loss = 0.0
            
            for batch_idx, (images, captions, labels) in enumerate(train_loader):
                images = images.to(self.device)
                
                # Label'ları tensor'e çevir
                batch_labels = []
                for label_dict in labels:
                    label_tensor = torch.tensor([
                        label_dict['violence'],
                        label_dict['adult_content'],
                        label_dict['harassment'],
                        label_dict['weapon'],
                        label_dict['drug'],
                        label_dict['safe']
                    ], dtype=torch.float32)
                    batch_labels.append(label_tensor)
                
                batch_labels = torch.stack(batch_labels).to(self.device)
                
                optimizer.zero_grad()
                
                # CLIP image features (frozen)
                with torch.no_grad():
                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Classification head
                predictions = self.classification_head(image_features)
                
                # Loss hesapla
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader, criterion)
            
            # History güncelle
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # En iyi modeli kaydet
                self.best_model_state = self.classification_head.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # En iyi modeli yükle
        if hasattr(self, 'best_model_state'):
            self.classification_head.load_state_dict(self.best_model_state)
        
        return history
    
    def _validate(self, val_loader, criterion) -> Tuple[float, float]:
        """Validation phase"""
        self.classification_head.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, captions, labels in val_loader:
                images = images.to(self.device)
                
                # Label'ları tensor'e çevir
                batch_labels = []
                for label_dict in labels:
                    label_tensor = torch.tensor([
                        label_dict['violence'],
                        label_dict['adult_content'],
                        label_dict['harassment'],
                        label_dict['weapon'],
                        label_dict['drug'],
                        label_dict['safe']
                    ], dtype=torch.float32)
                    batch_labels.append(label_tensor)
                
                batch_labels = torch.stack(batch_labels).to(self.device)
                
                # CLIP image features
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Predictions
                predictions = self.classification_head(image_features)
                
                # Loss
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()
                
                # Accuracy (threshold 0.5)
                pred_binary = (predictions > 0.5).float()
                correct_predictions += (pred_binary == batch_labels).all(dim=1).sum().item()
                total_predictions += batch_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_val_loss, val_accuracy
    
    def _save_trained_model(self, training_params: Dict) -> str:
        """Eğitilmiş modeli kaydet"""
        try:
            # Model klasörü oluştur
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_name = f"clip_finetuned_v{timestamp}"
            
            model_dir = os.path.join(
                current_app.config['OPENCLIP_MODEL_VERSIONS_PATH'],
                version_name
            )
            os.makedirs(model_dir, exist_ok=True)
            
            # CLIP base model + classification head kaydet
            model_data = {
                'clip_state_dict': self.model.state_dict(),
                'classification_head_state_dict': self.classification_head.state_dict(),
                'model_config': {
                    'clip_model_name': 'ViT-H-14-378-quickgelu',
                    'pretrained': 'dfn5b',
                    'num_classes': 6
                },
                'training_params': training_params,
                'timestamp': timestamp,
                'version_name': version_name
            }
            
            model_path = os.path.join(model_dir, 'open_clip_pytorch_model.bin')
            torch.save(model_data, model_path)
            
            logger.info(f"✅ Model kaydedildi: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
            raise e
    
    def _save_training_session(self, training_data: Dict, training_params: Dict, history: Dict, model_path: str) -> CLIPTrainingSession:
        """Training session'ı veritabanına kaydet"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_name = f"clip_finetuned_v{timestamp}"
            
            # Tüm mevcut session'ları pasif yap
            db.session.query(CLIPTrainingSession).update({'is_active': False})
            
            # Yeni session oluştur
            training_session = CLIPTrainingSession(
                version_name=version_name,
                feedback_count=training_data['total_samples'],
                training_start=datetime.now(),
                training_end=datetime.now(),
                status='completed',
                training_params=json.dumps(training_params),
                performance_metrics=json.dumps({
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'final_val_accuracy': history['val_accuracy'][-1],
                    'epochs_completed': len(history['train_loss']),
                    'train_samples': training_data['train_samples'],
                    'val_samples': training_data['val_samples']
                }),
                model_path=model_path,
                is_active=True,
                is_successful=True
            )
            
            db.session.add(training_session)
            db.session.commit()
            
            logger.info(f"✅ Training session kaydedildi: {training_session.id}")
            return training_session
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Training session kaydetme hatası: {e}")
            raise e
    
    def get_training_statistics(self) -> Dict:
        """Training istatistiklerini döndür"""
        try:
            # Toplam feedback sayısı
            total_feedback = db.session.query(Feedback).filter(
                Feedback.feedback_type == 'content'
            ).count()
            
            # Geçerli feedback sayısı (frame_path olan)
            valid_feedback = db.session.query(Feedback).filter(
                Feedback.feedback_type == 'content',
                Feedback.frame_path.isnot(None)
            ).count()
            
            # Training sessions
            total_sessions = db.session.query(CLIPTrainingSession).count()
            successful_sessions = db.session.query(CLIPTrainingSession).filter(
                CLIPTrainingSession.is_successful == True
            ).count()
            
            # Aktif session
            active_session = db.session.query(CLIPTrainingSession).filter(
                CLIPTrainingSession.is_active == True
            ).first()
            
            stats = {
                'total_content_feedback': total_feedback,
                'valid_content_feedback': valid_feedback,
                'total_training_sessions': total_sessions,
                'successful_training_sessions': successful_sessions,
                'active_session': {
                    'id': active_session.id if active_session else None,
                    'version_name': active_session.version_name if active_session else None,
                    'created_at': active_session.created_at.isoformat() if active_session and active_session.created_at else None
                } if active_session else None,
                'ready_for_training': valid_feedback >= 10
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Training statistics hatası: {e}")
            return {'error': str(e)} 