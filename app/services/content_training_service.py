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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import pickle

logger = logging.getLogger(__name__)

class ContentTrainingService:
    """OpenCLIP modeli için fine-tuning servisi - Geliştirilmiş Feedback Sistemi"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu"
        self.categories = ['violence', 'adult_content', 'harassment', 'weapon', 'drug']
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        logger.info(f"ContentTrainingService initialized with device: {self.device}")
    
    def load_base_model(self):
        """Base OpenCLIP modelini yükle"""
        try:
            clip_model_name = 'ViT-H-14-378-quickgelu'
            pretrained_weights_path = current_app.config['OPENCLIP_MODEL_BASE_PATH']
            
            if not os.path.exists(pretrained_weights_path):
                logger.error(f"Base model path not found: {pretrained_weights_path}")
                return False
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model_name, 
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
    
    def prepare_training_data(self, min_samples=50, validation_strategy='strict'):
        """
        Geliştirilmiş eğitim verisi hazırlama - Feedback verilerini düzgün işle
        
        Args:
            min_samples: Minimum gereken örnek sayısı
            validation_strategy: 'strict' (sadece tutarlı veriler), 'balanced' (dengelenmiş), 'all' (tüm veriler)
        """
        try:
            # DOĞRU feedback filtreleme - content feedback'leri al
            feedbacks = db.session.query(Feedback).filter(
                Feedback.feedback_source == 'MANUAL_USER',
                Feedback.category_feedback.isnot(None),  # JSON field boş değil
                Feedback.category_correct_values.isnot(None),  # JSON field boş değil
                Feedback.frame_path.isnot(None)
            ).all()
            
            logger.info(f"Found {len(feedbacks)} potential feedback records")
            
            if len(feedbacks) < min_samples:
                logger.warning(f"Insufficient feedback data: {len(feedbacks)} < {min_samples}")
                return None
            
            # Feedback'leri işle ve eğitim verisi oluştur
            training_samples = []
            category_stats = {cat: {'positive': 0, 'negative': 0, 'corrections': 0} for cat in self.categories}
            
            for feedback in feedbacks:
                if not feedback.frame_path or not os.path.exists(feedback.frame_path):
                    continue
                
                try:
                    # JSON verilerini parse et
                    category_feedback = feedback.category_feedback or {}
                    category_correct_values = feedback.category_correct_values or {}
                    
                    # Her kategori için veri işle
                    for category in self.categories:
                        if category not in category_feedback:
                            continue
                        
                        feedback_type = category_feedback[category]
                        if not feedback_type or feedback_type == "":
                            continue
                        
                        # Doğru skor belirle
                        target_score = self._calculate_target_score(
                            feedback_type, 
                            category_correct_values.get(category)
                        )
                        
                        if target_score is None:
                            continue
                        
                        # Validation strategy kontrolü
                        if not self._validate_sample(feedback_type, target_score, validation_strategy):
                            continue
                        
                        training_samples.append({
                            'image_path': feedback.frame_path,
                            'category': category,
                            'target_score': target_score,  # 0.0 - 1.0 arası
                            'feedback_type': feedback_type,
                            'feedback_id': feedback.id,
                            'original_score': category_correct_values.get(category, 0) / 100.0  # Normalize to 0-1
                        })
                        
                        # İstatistikleri güncelle
                        if target_score > 0.5:
                            category_stats[category]['positive'] += 1
                        else:
                            category_stats[category]['negative'] += 1
                        
                        if feedback_type in ['score_too_low', 'score_too_high']:
                            category_stats[category]['corrections'] += 1
                
                except Exception as e:
                    logger.warning(f"Error processing feedback {feedback.id}: {str(e)}")
                    continue
            
            if len(training_samples) < min_samples:
                logger.warning(f"Insufficient processed samples: {len(training_samples)} < {min_samples}")
                return None
            
            logger.info(f"Prepared {len(training_samples)} training samples")
            logger.info(f"Category statistics: {category_stats}")
            
            return {
                'data': training_samples,
                'category_stats': category_stats,
                'total_samples': len(training_samples),
                'feedbacks_processed': len(feedbacks)
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None
    
    def _calculate_target_score(self, feedback_type, user_score):
        """
        Feedback türüne göre hedef skor hesapla
        
        Args:
            feedback_type: str - Feedback türü
            user_score: int/float - Kullanıcının verdiği doğru skor (0-100)
        
        Returns:
            float: Hedef skor (0.0-1.0) veya None
        """
        if feedback_type == 'false_positive':
            return 0.0  # Aslında yoktu
        
        elif feedback_type == 'false_negative':
            # Kullanıcı skor verdiyse onu kullan, yoksa yüksek değer
            if user_score is not None and user_score > 0:
                return min(user_score / 100.0, 1.0)
            return 0.8  # Varsayılan olarak yüksek skor
        
        elif feedback_type == 'correct':
            return None  # Model zaten doğru, eğitim verisi olarak kullanma
        
        elif feedback_type in ['score_too_low', 'score_too_high']:
            # Kullanıcının verdiği doğru skoru kullan
            if user_score is not None:
                return max(0.0, min(user_score / 100.0, 1.0))
            return None
        
        return None
    
    def _validate_sample(self, feedback_type, target_score, strategy):
        """Validation strategy'ye göre örneği kabul et/reddet"""
        if strategy == 'strict':
            # Sadece kesin feedback'leri kabul et
            return feedback_type in ['false_positive', 'false_negative', 'score_too_low', 'score_too_high']
        
        elif strategy == 'balanced':
            # Dengeli veri seti için tüm türleri kabul et
            return target_score is not None
        
        elif strategy == 'all':
            # Tüm verileri kabul et
            return target_score is not None
        
        return False
    
    def detect_feedback_conflicts(self, training_data):
        """
        Çelişkili feedback'leri tespit et
        Örnek: Aynı resim için birisi %20 diğeri %80 demiş
        """
        conflicts = []
        image_feedback = {}
        
        for sample in training_data['data']:
            key = (sample['image_path'], sample['category'])
            if key not in image_feedback:
                image_feedback[key] = []
            image_feedback[key].append(sample)
        
        for key, samples in image_feedback.items():
            if len(samples) > 1:
                scores = [s['target_score'] for s in samples]
                if max(scores) - min(scores) > 0.3:  # %30'dan fazla fark
                    conflicts.append({
                        'image_path': key[0],
                        'category': key[1],
                        'scores': scores,
                        'feedback_types': [s['feedback_type'] for s in samples],
                        'feedback_ids': [s['feedback_id'] for s in samples]
                    })
        
        if conflicts:
            logger.warning(f"Detected {len(conflicts)} feedback conflicts")
            for conflict in conflicts[:5]:  # İlk 5'ini logla
                logger.warning(f"Conflict: {conflict}")
        
        return conflicts
    
    def create_classification_head(self, regression_mode=True):
        """
        CLIP feature'ları için classification/regression head oluştur
        
        Args:
            regression_mode: True ise regression (0-1 skor), False ise classification
        """
        feature_dim = self.model.visual.output_dim
        
        if regression_mode:
            # Regression için (0-1 arası skor tahmini)
            return nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.categories)),
                nn.Sigmoid()  # 0-1 arası skor için
            ).to(self.device)
        else:
            # Binary classification için
            return nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.categories)),
                nn.Sigmoid()  # Multi-label için
            ).to(self.device)
    
    def train_model(self, training_data, params):
        """
        Geliştirilmiş model eğitimi - Regression ve conflict detection ile
        """
        try:
            if not self.load_base_model():
                raise Exception("Base model loading failed")
            
            # Çelişkili feedback'leri tespit et
            conflicts = self.detect_feedback_conflicts(training_data)
            
            # Training mode belirle
            regression_mode = params.get('regression_mode', True)
            conflict_resolution = params.get('conflict_resolution', 'average')  # 'average', 'latest', 'ignore'
            
            # Çelişkileri çöz
            cleaned_data = self._resolve_conflicts(training_data, conflict_resolution)
            
            # Classification/Regression head oluştur
            classifier = self.create_classification_head(regression_mode=regression_mode)
            
            # Dataset ve DataLoader oluştur
            dataset = ContentTrainingDataset(
                cleaned_data['data'], 
                self.preprocess, 
                self.categories,
                regression_mode=regression_mode
            )
            
            # Train/validation split
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size', 16), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params.get('batch_size', 16), shuffle=False)
            
            # Loss function ve optimizer
            if regression_mode:
                criterion = nn.MSELoss()  # Regression için MSE
            else:
                criterion = nn.BCELoss()  # Classification için BCE
                
            optimizer = optim.Adam(classifier.parameters(), lr=params.get('learning_rate', 0.001))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
            
            # CLIP modelini freeze et (sadece classification head'i eğit)
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Training loop
            num_epochs = params.get('epochs', 20)
            train_losses = []
            val_losses = []
            mae_scores = []
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = params.get('patience', 5)
            
            logger.info(f"Starting training: {num_epochs} epochs, {len(train_loader)} batches")
            logger.info(f"Mode: {'Regression' if regression_mode else 'Classification'}")
            logger.info(f"Conflicts detected: {len(conflicts)}, Resolution: {conflict_resolution}")
            
            for epoch in range(num_epochs):
                # Training phase
                classifier.train()
                train_loss = 0.0
                train_samples = 0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # CLIP feature extraction (frozen)
                    with torch.no_grad():
                        features = self.model.encode_image(images)
                        features = features / features.norm(dim=-1, keepdim=True)
                    
                    # Classification/Regression
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_samples += images.size(0)
                    
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
                scheduler.step(avg_val_loss)
                
                # Calculate metrics
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                
                if regression_mode:
                    # Regression metrikleri
                    mae = mean_absolute_error(all_labels.flatten(), all_preds.flatten())
                    mae_scores.append(mae)
                    
                    # R² score hesapla
                    ss_res = np.sum((all_labels.flatten() - all_preds.flatten()) ** 2)
                    ss_tot = np.sum((all_labels.flatten() - np.mean(all_labels.flatten())) ** 2)
                    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                    logger.info(f"MAE: {mae:.4f}, R²: {r2_score:.4f}")
                    
                    # WebSocket progress gönder (regression mode)
                    if 'session_id' in params and params['session_id']:
                        try:
                            from app import socketio
                            socketio.emit('training_progress', {
                                'session_id': params['session_id'],
                                'current_epoch': epoch + 1,
                                'total_epochs': num_epochs,
                                'current_loss': float(avg_val_loss),
                                'current_mae': float(mae),
                                'current_r2': float(r2_score)
                            })
                        except Exception as ws_error:
                            logger.warning(f"WebSocket emit error: {str(ws_error)}")
                    
                else:
                    # Classification metrikleri
                    binary_preds = (all_preds > 0.5).astype(int)
                    accuracy = accuracy_score(all_labels.flatten(), binary_preds.flatten())
                    precision = precision_score(all_labels.flatten(), binary_preds.flatten(), average='weighted', zero_division=0)
                    recall = recall_score(all_labels.flatten(), binary_preds.flatten(), average='weighted', zero_division=0)
                    f1 = f1_score(all_labels.flatten(), binary_preds.flatten(), average='weighted', zero_division=0)
                    
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                    logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    
                    # WebSocket progress gönder (classification mode)
                    if 'session_id' in params and params['session_id']:
                        try:
                            from app import socketio
                            socketio.emit('training_progress', {
                                'session_id': params['session_id'],
                                'current_epoch': epoch + 1,
                                'total_epochs': num_epochs,
                                'current_loss': float(avg_val_loss),
                                'current_accuracy': float(accuracy)
                            })
                        except Exception as ws_error:
                            logger.warning(f"WebSocket emit error: {str(ws_error)}")
                
                # Early stopping ve best model kaydetme
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(classifier.state_dict(), 'best_classifier.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            classifier.load_state_dict(torch.load('best_classifier.pth'))
            
            # Final metrics
            if regression_mode:
                final_metrics = {
                    'mae': float(mae),
                    'r2_score': float(r2_score),
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'mae_scores': mae_scores,
                    'best_val_loss': float(best_val_loss)
                }
            else:
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
                'conflicts_detected': len(conflicts),
                'conflicts_resolved': conflict_resolution,
                'regression_mode': regression_mode,
                'history': {
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'mae_scores': mae_scores if regression_mode else []
                },
                'used_feedback_ids': [item['feedback_id'] for item in cleaned_data['data']]
            }
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
    
    def _resolve_conflicts(self, training_data, resolution_strategy):
        """Çelişkili feedback'leri çöz"""
        conflicts = self.detect_feedback_conflicts(training_data)
        if not conflicts:
            return training_data
        
        cleaned_data = []
        conflict_keys = set()
        
        # Çelişkili örneklerin key'lerini topla
        for conflict in conflicts:
            conflict_keys.add((conflict['image_path'], conflict['category']))
        
        # Çelişkili olmayan örnekleri direkt ekle
        for sample in training_data['data']:
            key = (sample['image_path'], sample['category'])
            if key not in conflict_keys:
                cleaned_data.append(sample)
        
        # Çelişkili örnekleri çöz
        for conflict in conflicts:
            key = (conflict['image_path'], conflict['category'])
            
            # Bu conflict için tüm sample'ları bul
            conflict_samples = [s for s in training_data['data'] 
                             if (s['image_path'], s['category']) == key]
            
            if resolution_strategy == 'average':
                # Skorları ortala
                avg_score = np.mean([s['target_score'] for s in conflict_samples])
                resolved_sample = conflict_samples[0].copy()
                resolved_sample['target_score'] = avg_score
                resolved_sample['feedback_type'] = 'conflict_resolved_average'
                cleaned_data.append(resolved_sample)
                
            elif resolution_strategy == 'latest':
                # En son feedback'i kullan (en yüksek ID)
                latest_sample = max(conflict_samples, key=lambda x: x['feedback_id'])
                latest_sample['feedback_type'] += '_conflict_resolved_latest'
                cleaned_data.append(latest_sample)
                
            elif resolution_strategy == 'ignore':
                # Çelişkili örnekleri tamamen görmezden gel
                pass
        
        logger.info(f"Conflict resolution: {len(training_data['data'])} -> {len(cleaned_data)} samples")
        
        return {
            'data': cleaned_data,
            'category_stats': training_data['category_stats'],
            'total_samples': len(cleaned_data),
            'conflicts_resolved': len(conflicts)
        }
    
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
            
            # Base CLIP modelini doğru şekilde kaydet (HuggingFace checkpoint sorununu atla)
            try:
                # Option 1: Doğru pretrained modelini kaydet
                logger.info("Doğru DFN5B CLIP modeli kaydediliyor...")
                import open_clip
                
                temp_model, _, _ = open_clip.create_model_and_transforms(
                    model_name="ViT-H-14-378-quickgelu",
                    pretrained="dfn5b",  # Doğru pretrained
                    device='cpu'
                )
                
                # Doğru OpenCLIP state_dict'i kaydet
                version_model_path = os.path.join(version_path, 'open_clip_pytorch_model.bin')
                torch.save(temp_model.state_dict(), version_model_path)
                logger.info(f"✅ Doğru CLIP modeli kaydedildi: {version_model_path}")
                
            except Exception as clip_save_error:
                logger.error(f"CLIP model kaydetme hatası: {clip_save_error}")
                # Fallback: Base dosyayı kopyala (eski davranış)
                base_model_path = os.path.join(
                    current_app.config['OPENCLIP_MODEL_BASE_PATH'],
                    'open_clip_pytorch_model.bin'
                )
                version_model_path = os.path.join(version_path, 'open_clip_pytorch_model.bin')
                shutil.copy2(base_model_path, version_model_path)
                logger.warning("⚠️ Base model kopyalandı - bu HuggingFace formatında olabilir")
            
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
    """OpenCLIP content training için geliştirilmiş dataset"""
    
    def __init__(self, data, preprocess, categories, regression_mode=True):
        self.data = data
        self.preprocess = preprocess
        self.categories = categories
        self.regression_mode = regression_mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Resmi yükle ve preprocess et
            image = Image.open(item['image_path']).convert('RGB')
            image = self.preprocess(image)
            
            # Multi-label için label vector oluştur
            labels = torch.zeros(len(self.categories))
            if item['category'] in self.categories:
                cat_idx = self.categories.index(item['category'])
                
                if self.regression_mode:
                    # Regression mode: 0.0-1.0 arası skorlar
                    labels[cat_idx] = float(item['target_score'])
                else:
                    # Classification mode: binary labels
                    labels[cat_idx] = 1.0 if item['target_score'] > 0.5 else 0.0
            
            return image, labels
            
        except Exception as e:
            # Hatalı resim durumunda siyah resim döndür
            logger.warning(f"Error loading image {item['image_path']}: {str(e)}")
            image = torch.zeros(3, 224, 224)  # Siyah resim
            labels = torch.zeros(len(self.categories))
            return image, labels 