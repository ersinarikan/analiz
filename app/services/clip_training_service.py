import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import shutil
from datetime import datetime
from flask import current_app
from app.models.feedback import Feedback
from app.models.analysis import Analysis
from app.models.clip_training import CLIPTrainingSession
from app import db
import logging
from PIL import Image
import open_clip
import threading
import time

class CLIPContrastiveDataset(Dataset):
    """CLIP Contrastive Learning Dataset"""
    
    def __init__(self, pairs, preprocess):
        self.pairs = pairs
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        try:
            # Resmi yükle ve preprocess et
            image = Image.open(pair['image_path']).convert('RGB')
            image = self.preprocess(image)
            
            # Text'i tokenize et (open_clip otomatik yapar)
            text = pair['text']
            label = torch.tensor(pair['label'], dtype=torch.float32)
            
            return {
                'image': image,
                'text': text,
                'label': label,
                'category': pair['category']
            }
        except Exception as e:
            # Hatalı resim durumunda dummy data döndür
            dummy_image = torch.zeros(3, 224, 224)
            return {
                'image': dummy_image,
                'text': "dummy text",
                'label': torch.tensor(0.0),
                'category': 'unknown'
            }

class CLIPTrainingService:
    """CLIP Contrastive Learning Training Service"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clip_models_path = os.path.join(current_app.config['MODELS_FOLDER'], 'clip')
        self.base_model_path = os.path.join(self.clip_models_path, 'ViT-H-14-378-quickgelu_dfn5b', 'base_model')
        self.active_model_path = os.path.join(self.clip_models_path, 'ViT-H-14-378-quickgelu_dfn5b', 'active_model')
        self.versions_path = os.path.join(self.clip_models_path, 'versions')
        
        # Versions klasörünü oluştur
        os.makedirs(self.versions_path, exist_ok=True)
        
    def start_training_async(self, training_params):
        """Asenkron training başlat"""
        def training_thread():
            try:
                # Flask app'i import et ve context oluştur
                from app import create_app
                app = create_app('development')
                
                with app.app_context():
                    self.run_contrastive_training(training_params)
            except Exception as e:
                self.logger.error(f"Training thread hatası: {str(e)}")
                
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
        
    def run_contrastive_training(self, training_params):
        """Ana contrastive learning training fonksiyonu"""
        session_id = None
        session = None
        
        try:
            self.logger.info("CLIP Contrastive Learning başlatılıyor...")
            
            # 1. Feedback'leri al
            feedbacks, error = self.get_available_feedbacks(
                min_feedback_count=training_params.get('min_feedback_count', 50)
            )
            if error:
                raise Exception(f"Feedback alma hatası: {error}")
            
            # 2. Contrastive pairs oluştur
            pairs, error = self.create_contrastive_pairs(
                feedbacks, 
                categories=training_params.get('categories', ['violence', 'adult_content', 'harassment', 'weapon', 'drug'])
            )
            if error:
                raise Exception(f"Contrastive pairs oluşturma hatası: {error}")
            
            if len(pairs) < 10:
                raise Exception(f"Yetersiz training pair: {len(pairs)}")
            
            # 3. Training data hazırla
            data_info, error = self.prepare_training_data(
                pairs, 
                train_split=training_params.get('train_split', 0.8)
            )
            if error:
                raise Exception(f"Training data hazırlama hatası: {error}")
            
            # 4. Database session oluştur
            session = CLIPTrainingSession(
                version_name=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                feedback_count=len(feedbacks),
                status='training'
            )
            # Training params'ı JSON string olarak kaydet
            session.set_training_params(training_params)
            
            db.session.add(session)
            db.session.commit()
            session_id = session.id
            
            self.logger.info(f"Training session oluşturuldu: {session_id}")
            
            # 5. Model ve preprocess yükle
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Device: {device}")
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14-378-quickgelu',
                pretrained='dfn5b'
            )
            model = model.to(device)
            tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
            
            # 6. Dataset ve DataLoader oluştur
            train_dataset = CLIPContrastiveDataset(data_info['train_pairs'], preprocess)
            val_dataset = CLIPContrastiveDataset(data_info['val_pairs'], preprocess)
            
            batch_size = training_params.get('batch_size', 16)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # 7. Optimizer ve scheduler
            learning_rate = training_params.get('learning_rate', 1e-5)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            
            epochs = training_params.get('epochs', 5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # 8. Training loop
            best_val_loss = float('inf')
            training_history = []
            
            for epoch in range(epochs):
                self.logger.info(f"Epoch {epoch+1}/{epochs} başlıyor...")
                
                # Training
                train_loss = self._train_epoch(model, train_loader, optimizer, tokenizer, device)
                
                # Validation
                val_loss = self._validate_epoch(model, val_loader, tokenizer, device)
                
                # Scheduler step
                scheduler.step()
                
                # History kaydet
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'timestamp': datetime.now().isoformat()
                }
                training_history.append(epoch_info)
                
                self.logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # En iyi modeli kaydet
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model_checkpoint(model, session.version_name, epoch_info)
                
                # Session güncelle
                session.performance_metrics = {
                    'training_history': training_history,
                    'best_val_loss': best_val_loss,
                    'current_epoch': epoch + 1,
                    'total_epochs': epochs
                }
                db.session.commit()
            
            # 9. Training tamamlandı
            session.status = 'completed'
            session.is_successful = True
            session.model_path = os.path.join(self.versions_path, session.version_name)
            db.session.commit()
            
            self.logger.info(f"CLIP Fine-tuning tamamlandı! Session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Training hatası: {str(e)}")
            
            # Session'ı failed olarak işaretle
            if session:
                session.status = 'failed'
                session.is_successful = False
                session.performance_metrics = {'error': str(e)}
                db.session.commit()
            
            raise e
    
    def _train_epoch(self, model, train_loader, optimizer, tokenizer, device):
        """Bir epoch training"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            try:
                images = batch['image'].to(device)
                texts = batch['text']
                labels = batch['label'].to(device)
                
                # Text tokenize
                text_tokens = tokenizer(texts).to(device)
                
                # Forward pass
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Contrastive loss hesapla
                # Cosine similarity
                similarities = torch.mm(image_features, text_features.t())
                
                # Diagonal elements (matching pairs)
                positive_similarities = torch.diag(similarities)
                
                # Labels'a göre loss hesapla
                # Positive pairs için similarity yüksek olmalı
                # Negative pairs için similarity düşük olmalı
                target_similarities = labels * 2 - 1  # [0,1] -> [-1,1]
                
                loss = F.mse_loss(positive_similarities, target_similarities)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                self.logger.warning(f"Batch training hatası: {str(e)}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, model, val_loader, tokenizer, device):
        """Bir epoch validation"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    images = batch['image'].to(device)
                    texts = batch['text']
                    labels = batch['label'].to(device)
                    
                    # Text tokenize
                    text_tokens = tokenizer(texts).to(device)
                    
                    # Forward pass
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(text_tokens)
                    
                    # Normalize features
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    
                    # Contrastive loss hesapla
                    similarities = torch.mm(image_features, text_features.t())
                    positive_similarities = torch.diag(similarities)
                    target_similarities = labels * 2 - 1
                    
                    loss = F.mse_loss(positive_similarities, target_similarities)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.warning(f"Batch validation hatası: {str(e)}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def _save_model_checkpoint(self, model, version_name, epoch_info):
        """Model checkpoint kaydet"""
        try:
            checkpoint_dir = os.path.join(self.versions_path, version_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Model state dict kaydet
            model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
            torch.save(model.state_dict(), model_path)
            
            # Metadata kaydet
            metadata = {
                'version_name': version_name,
                'model_type': 'ViT-H-14-378-quickgelu',
                'pretrained': 'dfn5b',
                'epoch_info': epoch_info,
                'saved_at': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model checkpoint kaydedildi: {checkpoint_dir}")
            
        except Exception as e:
            self.logger.error(f"Model checkpoint kaydetme hatası: {str(e)}")
    
    def get_available_feedbacks(self, min_feedback_count=50):
        """Eğitim için kullanılabilir feedback'leri getir"""
        try:
            # Manuel feedback'leri al (daha güvenilir)
            manual_feedbacks = Feedback.query.filter(
                Feedback.feedback_type == 'manual',
                Feedback.category_feedback.isnot(None)
            ).all()
            
            # Pseudo-label feedback'leri al
            pseudo_feedbacks = Feedback.query.filter(
                Feedback.feedback_type == 'pseudo_label',
                Feedback.category_feedback.isnot(None)
            ).all()
            
            total_feedbacks = len(manual_feedbacks) + len(pseudo_feedbacks)
            
            self.logger.info(f"Toplam feedback: {total_feedbacks} (Manuel: {len(manual_feedbacks)}, Pseudo: {len(pseudo_feedbacks)})")
            
            if total_feedbacks < min_feedback_count:
                return None, f"Minimum {min_feedback_count} feedback gerekli, mevcut: {total_feedbacks}"
                
            return manual_feedbacks + pseudo_feedbacks, None
            
        except Exception as e:
            self.logger.error(f"Feedback'ler alınırken hata: {str(e)}")
            return None, str(e)
    
    def create_contrastive_pairs(self, feedbacks, categories=['violence', 'adult_content', 'harassment', 'weapon', 'drug']):
        """Feedback'lerden contrastive learning çiftleri oluştur"""
        pairs = []
        
        try:
            for feedback in feedbacks:
                if not feedback.category_feedback:
                    continue
                    
                image_path = feedback.frame_path
                if not os.path.exists(image_path):
                    continue
                
                for category in categories:
                    if category not in feedback.category_feedback:
                        continue
                        
                    user_score = feedback.category_feedback[category].get('user_score', 0.5)
                    
                    # Pozitif çiftler (yüksek skor = bu kategori var)
                    if user_score > 0.6:
                        pairs.append({
                            'image_path': image_path,
                            'text': self._get_positive_prompt(category),
                            'label': 1,
                            'category': category,
                            'score': user_score,
                            'feedback_id': feedback.id
                        })
                        
                    # Negatif çiftler (düşük skor = bu kategori yok)
                    elif user_score < 0.4:
                        pairs.append({
                            'image_path': image_path,
                            'text': self._get_negative_prompt(category),
                            'label': 0,
                            'category': category,
                            'score': user_score,
                            'feedback_id': feedback.id
                        })
            
            self.logger.info(f"Toplam {len(pairs)} contrastive pair oluşturuldu")
            return pairs, None
            
        except Exception as e:
            self.logger.error(f"Contrastive pairs oluşturulurken hata: {str(e)}")
            return [], str(e)
    
    def _get_positive_prompt(self, category):
        """Pozitif prompt'ları getir"""
        prompts = {
            'violence': "This image contains violence and aggression",
            'adult_content': "This image contains adult and explicit content", 
            'harassment': "This image shows harassment and intimidation",
            'weapon': "This image contains weapons and armament",
            'drug': "This image shows drug use and substance abuse"
        }
        return prompts.get(category, f"This image contains {category}")
    
    def _get_negative_prompt(self, category):
        """Negatif prompt'ları getir"""
        prompts = {
            'violence': "This image is peaceful and non-violent",
            'adult_content': "This image is appropriate and family-friendly",
            'harassment': "This image shows peaceful and respectful interaction", 
            'weapon': "This image is weapon-free and safe",
            'drug': "This image is drug-free and shows healthy behavior"
        }
        return prompts.get(category, f"This image is free from {category}")
    
    def prepare_training_data(self, pairs, train_split=0.8):
        """Training data'yı hazırla"""
        try:
            # Shuffle pairs
            np.random.shuffle(pairs)
            
            # Train/validation split
            split_idx = int(len(pairs) * train_split)
            train_pairs = pairs[:split_idx]
            val_pairs = pairs[split_idx:]
            
            # Kategori dağılımını analiz et
            category_stats = {}
            for pair in pairs:
                cat = pair['category']
                if cat not in category_stats:
                    category_stats[cat] = {'positive': 0, 'negative': 0}
                
                if pair['label'] == 1:
                    category_stats[cat]['positive'] += 1
                else:
                    category_stats[cat]['negative'] += 1
            
            self.logger.info(f"Training data hazırlandı: {len(train_pairs)} train, {len(val_pairs)} validation")
            self.logger.info(f"Kategori dağılımı: {category_stats}")
            
            return {
                'train_pairs': train_pairs,
                'val_pairs': val_pairs,
                'category_stats': category_stats,
                'total_pairs': len(pairs)
            }, None
            
        except Exception as e:
            self.logger.error(f"Training data hazırlanırken hata: {str(e)}")
            return None, str(e)
    
    def create_training_session(self, feedback_count, training_params):
        """Yeni training session oluştur"""
        try:
            session_id = f"clip_contrastive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version_name = f"v{datetime.now().strftime('%Y%m%d')}_{feedback_count}fb"
            
            session_data = {
                'session_id': session_id,
                'version_name': version_name,
                'feedback_count': feedback_count,
                'training_params': training_params,
                'created_at': datetime.now().isoformat(),
                'status': 'preparing'
            }
            
            # Session dosyasını kaydet
            session_file = os.path.join(self.versions_path, f"{session_id}.json")
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Training session oluşturuldu: {session_id}")
            return session_id, version_name, None
            
        except Exception as e:
            self.logger.error(f"Training session oluşturulurken hata: {str(e)}")
            return None, None, str(e)
    
    def get_training_statistics(self):
        """Training için istatistikleri getir"""
        try:
            feedbacks, error = self.get_available_feedbacks(min_feedback_count=1)
            if error:
                return None, error
            
            stats = {
                'total_feedbacks': len(feedbacks),
                'manual_feedbacks': len([f for f in feedbacks if f.feedback_type == 'manual']),
                'pseudo_feedbacks': len([f for f in feedbacks if f.feedback_type == 'pseudo_label']),
                'category_distribution': {},
                'ready_for_training': len(feedbacks) >= 50
            }
            
            # Kategori dağılımını hesapla
            categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
            for category in categories:
                stats['category_distribution'][category] = {
                    'high_score': 0,  # >0.6
                    'medium_score': 0,  # 0.4-0.6
                    'low_score': 0   # <0.4
                }
                
                for feedback in feedbacks:
                    if not feedback.category_feedback or category not in feedback.category_feedback:
                        continue
                        
                    user_score = feedback.category_feedback[category].get('user_score', 0.5)
                    if user_score > 0.6:
                        stats['category_distribution'][category]['high_score'] += 1
                    elif user_score < 0.4:
                        stats['category_distribution'][category]['low_score'] += 1
                    else:
                        stats['category_distribution'][category]['medium_score'] += 1
            
            return stats, None
            
        except Exception as e:
            self.logger.error(f"Training istatistikleri alınırken hata: {str(e)}")
            return None, str(e) 