import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from flask import current_app
from app import db
from app.models.feedback import Feedback
from app.models.content import ModelVersion
from app.ai.insightface_age_estimator import CustomAgeHead
from config import Config
from app.utils.model_utils import save_torch_model
from app.utils.file_utils import ensure_dir, safe_copytree, safe_remove, write_json, read_json, get_folder_size

# Root logger'ı kullan (terminalde görünmesi için)
logger = logging.getLogger('app.age_training')

class AgeTrainingService:
    """
    Yaş tahmini modelinin eğitimini ve veri hazırlığını yöneten servis sınıfı.
    - Geri bildirim verisi toplama, eğitim ve temizlik işlemlerini içerir.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        logger.info(f"AgeTrainingService initialized with device: {self.device}")
    
    def prepare_training_data(self, min_samples: int = 10) -> dict | None:
        """
        Feedback tablosundan yaş eğitim verilerini hazırlar
        
        Args:
            min_samples: Minimum gerekli örnek sayısı
            
        Returns:
            dict: Eğitim verisi (embeddings, ages, sources, confidence_scores, feedback_ids)
            None: Yetersiz veri durumunda
        """
        logger.info("Preparing training data from feedback table...")
        
        # Yaş geri bildirimi olan ve daha önce eğitimde kullanılmamış kayıtları al
        feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).filter(
            Feedback.embedding.isnot(None)
        ).filter(
            # Daha önce eğitimde kullanılmamış verileri al
            db.or_(
                Feedback.training_status.is_(None),
                Feedback.training_status != 'used_in_training'
            )
        ).all()
        
        logger.info(f"Found {len(feedbacks)} unused age feedback records with embeddings")
        
        if len(feedbacks) < min_samples:
            logger.warning(f"Insufficient unused feedback data: {len(feedbacks)} < {min_samples}")
            # Eğer yeterli yeni veri yoksa, kullanılmış verileri de dahil et (opsiyonel)
            logger.info("Checking if we should include previously used data...")
            
            all_feedbacks = Feedback.query.filter(
                (Feedback.feedback_type == 'age') | 
                (Feedback.feedback_type == 'age_pseudo')
            ).filter(
                Feedback.embedding.isnot(None)
            ).all()
            
            logger.info(f"Total available feedback records: {len(all_feedbacks)}")
            
            if len(all_feedbacks) >= min_samples:
                logger.warning("Using previously used training data due to insufficient new data")
                feedbacks = all_feedbacks
            else:
                logger.error(f"Insufficient total feedback data: {len(all_feedbacks)} < {min_samples}")
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
        """
        if params is None:
            params = Config.DEFAULT_TRAINING_PARAMS.copy()
        else:
            default_params = Config.DEFAULT_TRAINING_PARAMS.copy()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
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
        
        # KRİTİK: EMBEDDING NORMALİZASYONU (inference ile tutarlılık için)
        X_train_tensor = X_train_tensor / torch.norm(X_train_tensor, dim=1, keepdim=True)
        X_val_tensor = X_val_tensor / torch.norm(X_val_tensor, dim=1, keepdim=True)
        logger.info("Embeddings normalized during training (to match inference normalization)")
        
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
            
            # Progress callback çağır (eğer varsa)
            if 'progress_callback' in params and callable(params['progress_callback']):
                try:
                    current_metrics = {
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'val_mae': avg_val_mae
                    }
                    params['progress_callback'](epoch + 1, params['epochs'], current_metrics)
                except Exception as e:
                    logger.warning(f"Progress callback error: {str(e)}")
            
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
        """
        # Versiyon numarasını belirle
        last_version = ModelVersion.query.filter_by(
            model_type='age'
        ).order_by(ModelVersion.version.desc()).first()
        new_version_num = 1 if last_version is None else last_version.version + 1
        if version_name is None:
            version_name = f"v{new_version_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = os.path.join(
            current_app.config['MODELS_FOLDER'],
            'age',
            'custom_age_head',
            'versions',
            version_name
        )
        # Model ağırlıklarını ve metadata'yı kaydet
        config_dict = {
            'input_dim': model.network[0].in_features,
            'hidden_dims': [layer.out_features for layer in model.network if hasattr(layer, 'out_features')][:-1],
            'output_dim': 1
        }
        extra_metadata = {
            'version': new_version_num,
            'version_name': version_name,
            'created_at': datetime.now().isoformat(),
            'model_type': 'age',
            'metrics': training_result['metrics'],
            'training_samples': training_result['training_samples'],
            'validation_samples': training_result['validation_samples'],
            'used_feedback_ids': training_result['used_feedback_ids']
        }
        model_path = save_torch_model(model, version_dir, config_dict, extra_metadata)
        # Eğitim detaylarını kaydet
        details_path = os.path.join(version_dir, 'training_details.json')
        with open(details_path, 'w') as f:
            json.dump(training_result, f, indent=4, default=str)
        # Veritabanına kaydet
        model_version = ModelVersion(
            model_type='age',
            version=new_version_num,
            version_name=version_name,
            created_at=datetime.now(),
            metrics=training_result['metrics'],
            is_active=False,
            training_samples=training_result['training_samples'],
            validation_samples=training_result['validation_samples'],
            epochs=len(training_result['history']['train_loss']),
            file_path=version_dir,
            weights_path=model_path,
            used_feedback_ids=training_result['used_feedback_ids']
        )
        db.session.add(model_version)
        db.session.commit()
        return model_version
    
    def activate_model_version(self, version_id):
        """
        Belirli bir model versiyonunu aktif hale getirir
        
        Args:
            version_id: Aktif edilecek ModelVersion ID'si veya 'base' (base model için)
            
        Returns:
            bool: Başarılı olup olmadığı
        """
        try:
            # Base model kontrolü
            if version_id == 'base':
                # Tüm versiyonları deaktive et
                ModelVersion.query.filter_by(
                    model_type='age',
                    is_active=True
                ).update({'is_active': False})
                db.session.commit()
                logger.info("Base model activated")
                return True
                
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
            
            # Yeni sembolik link oluştur veya kopyala
            if not version.file_path or not os.path.exists(version.file_path):
                logger.error(f"Aktif edilecek modelin dosya yolu bulunamadı veya mevcut değil: {version.file_path}")
                return False
            try:
                os.symlink(version.file_path, active_dir)
                logger.info(f"Sembolik link oluşturuldu: {active_dir} -> {version.file_path}")
            except Exception as symlink_err:
                import shutil
                logger.warning(f"Sembolik link oluşturulamadı ({symlink_err}), klasör kopyalanacak...")
                try:
                    shutil.copytree(version.file_path, active_dir)
                    logger.info(f"Model klasörü kopyalandı: {version.file_path} -> {active_dir}")
                except Exception as copy_err:
                    logger.error(f"Model klasörü kopyalanamadı: {copy_err}")
                    return False
            
            # Aktif edilen versiyonun bilgisini version_info.json'a yaz
            try:
                import json
                version_info = {'version_name': version.version_name}
                # 1) Symlink/kopya olan active_model dizinine yaz
                version_info_path = os.path.join(active_dir, 'version_info.json')
                with open(version_info_path, 'w') as f:
                    json.dump(version_info, f)
                logger.info(f"Aktif versiyon bilgisi yazıldı: {version_info_path} -> {version.version_name}")
                # 2) Hedef versiyon klasörüne de yaz (symlink ise zaten aynı yere yazar, kopya ise iki yerde de olur)
                if version.file_path and os.path.exists(version.file_path):
                    version_info_path_target = os.path.join(version.file_path, 'version_info.json')
                    with open(version_info_path_target, 'w') as f:
                        json.dump(version_info, f)
                    logger.info(f"Aktif versiyon bilgisi hedef klasöre de yazıldı: {version_info_path_target} -> {version.version_name}")
                # Ek kontrol: Dosyalar gerçekten oluştu mu?
                if not os.path.exists(version_info_path):
                    logger.error(f"version_info.json aktif_model dizininde bulunamadı: {version_info_path}")
                if version.file_path and not os.path.exists(os.path.join(version.file_path, 'version_info.json')):
                    logger.error(f"version_info.json hedef versiyon klasöründe bulunamadı: {version_info_path_target}")
            except Exception as info_err:
                logger.error(f"Aktif versiyon bilgisi yazılamadı: {info_err}")
            
            logger.info(f"Activated model version: {version.version_name} (ID: {version_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating model version: {str(e)}")
            db.session.rollback()
            return False
    
    def get_model_versions(self) -> list[dict]:
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

    def cleanup_training_data(self, dry_run: bool = True) -> dict:
        """
        Eğitim verilerini temizler (silmez, sadece işaretler)
        
        Args:
            dry_run: True ise sadece rapor verir, değişiklik yapmaz
            
        Returns:
            dict: Temizlik raporu
        """
        logger.info(f"Starting training data cleanup (dry_run={dry_run})")
        
        # Temizlik politikası
        policy = current_app.config.get('TRAINING_DATA_RETENTION_POLICY', {
            'pseudo_label_max_age_days': 180,
            'max_feedback_per_person': 3,
            'keep_manual_feedback': True
        })
        
        cleanup_report = {
            'total_feedbacks': 0,
            'duplicate_pseudo_labels': 0,
            'old_pseudo_labels': 0,
            'excess_feedbacks_per_person': 0,
            'invalid_data': 0,
            'actions_taken': []
        }
        
        # Tüm yaş feedback'lerini al
        all_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo')
        ).all()
        
        cleanup_report['total_feedbacks'] = len(all_feedbacks)
        
        # 1. Eski pseudo-label'ları bul
        cutoff_date = datetime.now() - timedelta(days=policy['pseudo_label_max_age_days'])
        old_pseudo_labels = Feedback.query.filter(
            Feedback.feedback_type == 'age_pseudo',
            Feedback.created_at < cutoff_date
        ).all()
        
        cleanup_report['old_pseudo_labels'] = len(old_pseudo_labels)
        
        if not dry_run:
            for feedback in old_pseudo_labels:
                feedback.is_archived = True
                feedback.archive_reason = f'old_pseudo_label_{policy["pseudo_label_max_age_days"]}_days'
                cleanup_report['actions_taken'].append(f'Archived old pseudo-label: {feedback.id}')
        
        # 2. Person başına fazla feedback'leri bul
        person_feedbacks = {}
        for feedback in all_feedbacks:
            if feedback.person_id:
                if feedback.person_id not in person_feedbacks:
                    person_feedbacks[feedback.person_id] = []
                person_feedbacks[feedback.person_id].append(feedback)
        
        excess_count = 0
        for person_id, feedbacks in person_feedbacks.items():
            if len(feedbacks) > policy['max_feedback_per_person']:
                # Manuel feedback'leri koru, pseudo'ları sırala
                manual_feedbacks = [f for f in feedbacks if f.feedback_source == 'MANUAL_USER']
                pseudo_feedbacks = [f for f in feedbacks if f.feedback_source != 'MANUAL_USER']
                
                # En yeni pseudo feedback'leri koru
                pseudo_feedbacks.sort(key=lambda x: x.created_at, reverse=True)
                
                # Fazla olanları işaretle
                keep_count = max(0, policy['max_feedback_per_person'] - len(manual_feedbacks))
                excess_pseudo = pseudo_feedbacks[keep_count:]
                
                excess_count += len(excess_pseudo)
                
                if not dry_run:
                    for feedback in excess_pseudo:
                        feedback.is_archived = True
                        feedback.archive_reason = f'excess_feedback_per_person_{policy["max_feedback_per_person"]}'
                        cleanup_report['actions_taken'].append(f'Archived excess feedback: {feedback.id}')
        
        cleanup_report['excess_feedbacks_per_person'] = excess_count
        
        # 3. Geçersiz veri kontrolü
        invalid_feedbacks = Feedback.query.filter(
            (Feedback.feedback_type == 'age') | 
            (Feedback.feedback_type == 'age_pseudo'),
            db.or_(
                Feedback.embedding.is_(None),
                Feedback.corrected_age < 0,
                Feedback.corrected_age > 120,
                Feedback.pseudo_label_original_age < 0,
                Feedback.pseudo_label_original_age > 120
            )
        ).all()
        
        cleanup_report['invalid_data'] = len(invalid_feedbacks)
        
        if not dry_run:
            for feedback in invalid_feedbacks:
                feedback.is_archived = True
                feedback.archive_reason = 'invalid_data'
                cleanup_report['actions_taken'].append(f'Archived invalid data: {feedback.id}')
            
            db.session.commit()
            logger.info("Training data cleanup completed and committed to database")
        else:
            logger.info("Training data cleanup completed (dry run - no changes made)")
        
        return cleanup_report

    def cleanup_used_training_data(self, used_feedback_ids: list[int], model_version_name: str) -> dict:
        """
        Eğitimde kullanılan verileri tamamen temizler (VT + dosyalar)
        
        Args:
            used_feedback_ids: Kullanılan feedback ID'leri
            model_version_name: Model versiyon adı
            
        Returns:
            dict: Temizlik raporu
        """
        logger.info(f"Cleaning up training data for model {model_version_name}")
        
        cleanup_report = {
            'deleted_feedbacks': 0,
            'deleted_files': 0,
            'deleted_directories': 0,
            'errors': []
        }
        
        try:
            # 1. Önce feedback'leri al (dosya yollarını almak için)
            feedbacks_to_delete = Feedback.query.filter(
                Feedback.id.in_(used_feedback_ids)
            ).all()
            
            # 2. İlgili dosya yollarını topla
            frame_paths = set()
            person_ids = set()
            
            for feedback in feedbacks_to_delete:
                if feedback.frame_path:
                    frame_paths.add(feedback.frame_path)
                if feedback.person_id:
                    person_ids.add(feedback.person_id)
            
            logger.info(f"Found {len(frame_paths)} frame paths and {len(person_ids)} person IDs to clean")
            
            # 3. Processed klasöründeki ilgili dosyaları sil
            processed_dir = current_app.config.get('PROCESSED_FOLDER', 'storage/processed')
            
            # Frame klasörlerini kontrol et ve sil
            if os.path.exists(processed_dir):
                for item in os.listdir(processed_dir):
                    item_path = os.path.join(processed_dir, item)
                    
                    if os.path.isdir(item_path) and item.startswith('frames_'):
                        # Bu frame klasöründe silinecek person_id'ler var mı kontrol et
                        should_delete_dir = False
                        
                        try:
                            # Klasör içindeki dosyaları kontrol et
                            for file_name in os.listdir(item_path):
                                # Person ID'li dosyaları kontrol et
                                for person_id in person_ids:
                                    if person_id in file_name:
                                        should_delete_dir = True
                                        break
                                if should_delete_dir:
                                    break
                            
                            # Eğer bu klasörde silinecek veriler varsa, tüm klasörü sil
                            if should_delete_dir:
                                import shutil
                                shutil.rmtree(item_path)
                                cleanup_report['deleted_directories'] += 1
                                logger.info(f"Deleted directory: {item_path}")
                                
                        except Exception as e:
                            logger.error(f"Error processing directory {item_path}: {str(e)}")
                            cleanup_report['errors'].append(f"Directory error: {str(e)}")
            
            # 4. Belirli frame dosyalarını sil (eğer tam yol verilmişse)
            for frame_path in frame_paths:
                try:
                    if frame_path and os.path.exists(frame_path):
                        os.remove(frame_path)
                        cleanup_report['deleted_files'] += 1
                        logger.info(f"Deleted file: {frame_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {frame_path}: {str(e)}")
                    cleanup_report['errors'].append(f"File deletion error: {str(e)}")
            
            # 5. Veritabanından feedback kayıtlarını sil
            deleted_feedbacks = Feedback.query.filter(
                Feedback.id.in_(used_feedback_ids)
            ).delete(synchronize_session=False)
            
            cleanup_report['deleted_feedbacks'] = deleted_feedbacks
            logger.info(f"Deleted {deleted_feedbacks} feedback records from database")
            
            db.session.commit()
            logger.info(f"Training data cleanup completed for model {model_version_name}")
            
        except Exception as e:
            logger.error(f"Error during training data cleanup: {str(e)}")
            cleanup_report['errors'].append(str(e))
            db.session.rollback()
            
        return cleanup_report

    def mark_training_data_used(self, feedback_ids: list[int], model_version_name: str):
        """
        Eğitimde kullanılan verileri işaretler
        
        Args:
            feedback_ids: Kullanılan feedback ID'leri
            model_version_name: Hangi model versiyonunda kullanıldığı
        """
        logger.info(f"Marking {len(feedback_ids)} feedback records as used in training")
        
        for feedback_id in feedback_ids:
            feedback = Feedback.query.get(feedback_id)
            if feedback:
                feedback.training_status = 'used_in_training'
                feedback.used_in_model_version = model_version_name
                feedback.training_used_at = datetime.now()
        
        db.session.commit()
        logger.info("Feedback records marked as used")

    def reset_to_base_model(self) -> bool:
        """
        Yaş tahmin modelini base (ön eğitimli) modele sıfırlar
        
        Returns:
            bool: Başarılı olup olmadığı
        """
        try:
            logger.info("Resetting age model to base model")
            
            # Aktif model dizinini kontrol et
            active_dir = os.path.join(
                current_app.config['MODELS_FOLDER'],
                'age',
                'custom_age_head',
                'active_model'
            )
            
            base_dir = os.path.join(
                current_app.config['MODELS_FOLDER'],
                'age',
                'custom_age_head',
                'base_model'
            )
            
            # Base model dizini var mı kontrol et
            if not os.path.exists(base_dir):
                logger.error(f"Custom age head base model directory not found: {base_dir}")
                logger.error("Custom age head base model is required for reset operation")
                logger.error("Buffalo_L model cannot be used as fallback for custom age head")
                return False
            
            # Önceki aktif modeli yedekle
            if os.path.exists(active_dir):
                backup_dir = os.path.join(
                    current_app.config['MODELS_FOLDER'],
                    'age',
                    'custom_age_head',
                    'backups',
                    f"active_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                if os.path.islink(active_dir):
                    # Sembolik link ise, hedef dizini yedekle
                    target_dir = os.readlink(active_dir)
                    if os.path.exists(target_dir):
                        import shutil
                        os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
                        shutil.copytree(target_dir, backup_dir)
                        logger.info(f"Backed up symbolic link target to: {backup_dir}")
                    os.unlink(active_dir)
                    logger.info("Removed existing symbolic link")
                else:
                    # Dizin ise, tüm dizini yedekle
                    import shutil
                    os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
                    shutil.move(active_dir, backup_dir)
                    logger.info(f"Backed up active model directory to: {backup_dir}")
            
            # Base modeli aktif model olarak ayarla
            if os.name == 'nt':  # Windows
                # Windows'ta sembolik link yerine dizini kopyala
                import shutil
                shutil.copytree(base_dir, active_dir)
                logger.info(f"Copied base model to active model (Windows)")
            else:  # Linux/Mac
                os.symlink(base_dir, active_dir)
                logger.info(f"Created symbolic link from base model to active model")
            
            logger.info("Age model successfully reset to base model")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting to base model: {str(e)}")
            return False 