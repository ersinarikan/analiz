import os
import json
import logging
import numpy as np
import torch
from datetime import datetime
from flask import current_app
from app import db
from app.models.feedback import Feedback
from app.models.content import ModelVersion

logger = logging.getLogger('app.ensemble_age_service')

class EnsembleAgeService:
    """
    Yaş tahmini için ensemble tabanlı artımsal öğrenme servis sınıfı.
    - Temel modeli korur, geri bildirim düzeltmelerini uygular, yaş benzerliği ile eşleştirme yapar.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and current_app.config.get('USE_GPU', True) else "cpu")
        self.feedback_corrections = {}  # person_id -> age_correction
        self.embedding_corrections = {}  # embedding_hash -> age_correction
        logger.info(f"EnsembleAgeService initialized")
    
    def _mark_feedback_as_used(self, feedback):
        """
        Feedback kaydını correctionda kullanıldı olarak işaretler.
        """
        feedback.training_status = 'used_in_correction'
        db.session.add(feedback)
        db.session.commit()
    
    def load_feedback_corrections(self) -> dict:
        """
        Veritabanından yaş düzeltmelerini yükle (Sadece embedding-based)
        """
        logger.info("Loading age feedback corrections from database (embedding-based only)...")
        
        try:
            # Yaş geri bildirimi olan kayıtları al (sadece embedding olanları)
            feedbacks = db.session.query(Feedback).filter(
                Feedback.corrected_age.isnot(None),
                Feedback.embedding.isnot(None)
            ).all()
            
            corrections_loaded = 0
            
            for feedback in feedbacks:
                try:
                    # Embedding'i parse et
                    embedding_str = feedback.embedding
                    if embedding_str:
                        embedding = np.array([float(x) for x in embedding_str.split(',')])
                        
                        # Sadece embedding düzeltmesi (person_id artık kullanılmıyor)
                        embedding_hash = self._hash_embedding(embedding)
                        self.embedding_corrections[embedding_hash] = {
                            'corrected_age': feedback.corrected_age,
                            'original_age': feedback.pseudo_label_original_age,
                            'embedding': embedding,
                            'confidence': 0.95,
                            'person_id': feedback.person_id,  # Sadece bilgi amaçlı (lookup'ta kullanılmıyor)
                            'source': feedback.feedback_source or 'MANUAL_USER',
                            'feedback_id': feedback.id,
                            'created_at': feedback.created_at.isoformat() if feedback.created_at else None
                        }
                        
                        corrections_loaded += 1
                        
                        # Feedback'i kullanım ile işaretle
                        self._mark_feedback_as_used(feedback)
                        
                except Exception as e:
                    logger.warning(f"Feedback parsing error for ID {feedback.id}: {str(e)}")
                    continue
            
            # Person ID tablosunu temizle (artık kullanılmıyor)
            self.feedback_corrections = {}
            
            # Değişiklikleri kaydet
            db.session.commit()
            
            logger.info(f"✅ {corrections_loaded} age correction loaded from database (embedding-based)")
            logger.info(f"📊 Total embedding corrections: {len(self.embedding_corrections)}")
            
            return corrections_loaded
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Error loading age feedback corrections: {str(e)}")
            return 0
    
    def _hash_embedding(self, embedding: np.ndarray) -> int:
        """Create hash for embedding (for lookup)"""
        # Simple hash based on first few dimensions
        return hash(tuple(embedding[:10].round(3)))
    
    def predict_age_ensemble(self, base_age: float, face_embedding: np.ndarray | None = None) -> tuple[float, float, dict]:
        """
        Yaş tahmini için ensemble yaklaşımı (Sadece embedding-based)
        1. Exact embedding match ile arama
        2. Embedding benzerliği ile arama  
        3. Son olarak base model sonucunu döndür
        """
        logger.info(f"🔍 Ensemble age prediction - Base: {base_age:.1f}")
        
        # Embedding yoksa base model kullan
        if face_embedding is None or len(self.embedding_corrections) == 0:
            logger.info(f"📊 No embedding or corrections available, using base model: {base_age:.1f}")
            return base_age, 0.7, {
                'method': 'base_model',
                'reason': 'no_embedding_or_corrections'
            }
        
        # 1. Exact embedding match
        embedding_hash = self._hash_embedding(face_embedding)
        if embedding_hash in self.embedding_corrections:
            correction = self.embedding_corrections[embedding_hash]
            corrected_age = correction['corrected_age']
            confidence = correction['confidence']
            
            logger.info(f"✅ Exact embedding match -> {corrected_age} years")
            
            # Kullanım sayısını artır
            if 'feedback_id' in correction:
                self._increment_usage_count(correction['feedback_id'])
            
            return corrected_age, confidence, {
                'method': 'exact_embedding_match',
                'person_id': correction.get('person_id'),
                'source': correction['source'],
                'feedback_id': correction.get('feedback_id')
            }
        
        # 2. Similarity-based correction
        best_similarity = -1
        best_correction = None
        
        # Normalize input embedding
        embedding_norm = face_embedding / np.linalg.norm(face_embedding)
        
        for emb_hash, correction in self.embedding_corrections.items():
            stored_embedding = correction['embedding']
            stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
            
            # Cosine similarity
            similarity = np.dot(embedding_norm, stored_embedding_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_correction = correction
        
        # Yüksek benzerlik varsa düzeltmeyi uygula (threshold lowered to 0.90 for better matching)
        if best_similarity > 0.90:  # %90 benzerlik eşiği (daha esnek)
            corrected_age = best_correction['corrected_age']
            confidence = best_correction['confidence'] * best_similarity  # Benzerlik oranında güven
            
            logger.info(f"✅ Similarity match ({best_similarity:.3f}) -> {corrected_age} years")
            
            # Kullanım sayısını artır
            if 'feedback_id' in best_correction:
                self._increment_usage_count(best_correction['feedback_id'])
            
            return corrected_age, confidence, {
                'method': 'similarity_match',
                'similarity': best_similarity,
                'person_id': best_correction.get('person_id'),
                'source': best_correction['source'],
                'feedback_id': best_correction.get('feedback_id')
            }
        
        # 3. Base model sonucunu döndür
        logger.info(f"📊 No similar embedding found (best: {best_similarity:.3f}), using base model: {base_age:.1f}")
        return base_age, 0.7, {
            'method': 'base_model',
            'reason': 'no_similar_embedding_found',
            'best_similarity': best_similarity
        }
    
    def _increment_usage_count(self, feedback_id: int):
        """Feedback kullanım sayısını artır"""
        try:
            feedback = db.session.query(Feedback).filter(Feedback.id == feedback_id).first()
            if feedback:
                feedback.ensemble_usage_count = (feedback.ensemble_usage_count or 0) + 1
                feedback.last_used_at = datetime.now()
                db.session.commit()
                logger.debug(f"Usage count incremented for feedback {feedback_id}")
        except Exception as e:
            logger.warning(f"Failed to increment usage count for feedback {feedback_id}: {str(e)}")
            db.session.rollback()
    
    def get_statistics(self) -> dict:
        """Get ensemble statistics (embedding-based only)"""
        return {
            'total_embedding_corrections': len(self.embedding_corrections),
            'total_people_corrections': len(self.embedding_corrections),  # Same as embedding corrections now
            'correction_method': 'embedding_based_only',
            'embedding_threshold': 0.90,
            'manual_corrections': len([c for c in self.embedding_corrections.values() if c['source'] == 'MANUAL_USER']),
            'pseudo_corrections': len([c for c in self.embedding_corrections.values() if 'PSEUDO' in c['source']])
        }
    
    def test_ensemble_predictions(self, test_cases=None):
        """Test ensemble on known feedback cases"""
        logger.info("Testing ensemble predictions...")
        
        if not self.feedback_corrections:
            logger.warning("No corrections loaded!")
            return
        
        # Test a few cases
        test_results = []
        
        for person_id, correction in list(self.feedback_corrections.items())[:5]:
            embedding = correction['embedding']
            corrected_age = correction['corrected_age']
            
            # Simulate base model prediction (add some noise)
            simulated_base_pred = corrected_age + np.random.normal(0, 2)  # +/- 2 years noise
            
            # Test ensemble
            ensemble_pred, confidence, info = self.predict_age_ensemble(
                simulated_base_pred, embedding
            )
            
            test_results.append({
                'person_id': person_id,
                'true_age': corrected_age,
                'base_pred': simulated_base_pred,
                'ensemble_pred': ensemble_pred,
                'confidence': confidence,
                'method': info['method']
            })
            
            logger.info(f"Test {person_id}: True={corrected_age:.1f}, "
                       f"Base={simulated_base_pred:.1f}, "
                       f"Ensemble={ensemble_pred:.1f}, "
                       f"Method={info['method']}")
        
        return test_results 
    
    def save_ensemble_corrections_as_version(self) -> ModelVersion:
        """
        Ensemble düzeltmelerini .pth dosyası olarak kaydet ve model versiyonu oluştur
        """
        logger.info("Saving ensemble corrections as model version...")
        
        try:
            # Versiyon numarasını belirle
            last_version = ModelVersion.query.filter_by(
                model_type='age'
            ).order_by(ModelVersion.version.desc()).first()
            
            new_version_num = 1 if last_version is None else last_version.version + 1
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_name = f"ensemble_v{new_version_num}_{timestamp}"
            
            # Versiyon klasörü oluştur
            version_dir = os.path.join(
                current_app.config['MODELS_FOLDER'],
                'age',
                'ensemble_versions',
                version_name
            )
            os.makedirs(version_dir, exist_ok=True)
            
            # Ensemble verilerini .pth formatında kaydet (sadece embedding-based)
            ensemble_data = {
                'model_type': 'ensemble_age',
                'embedding_corrections': self.embedding_corrections,  # Sadece embedding corrections
                'version': new_version_num,
                'version_name': version_name,
                'created_at': datetime.now().isoformat(),
                'total_corrections': len(self.embedding_corrections),  # Person ID artık yok
                'correction_method': 'embedding_based_only'
            }
            
            # .pth dosyası olarak kaydet
            model_path = os.path.join(version_dir, 'ensemble_corrections.pth')
            torch.save(ensemble_data, model_path)
            logger.info(f"Ensemble corrections saved to: {model_path}")
            
            # Metadata oluştur
            metadata = {
                'version': new_version_num,
                'version_name': version_name,
                'created_at': datetime.now().isoformat(),
                'model_type': 'ensemble_age',
                'total_corrections': len(self.embedding_corrections),
                'embedding_corrections': len(self.embedding_corrections),
                'correction_sources': self._get_correction_sources_stats(),
                'metrics': self._calculate_ensemble_metrics()
            }
            
            # Metadata kaydet
            metadata_path = os.path.join(version_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Tüm aktif versiyonları devre dışı bırak
            db.session.query(ModelVersion).filter_by(
                model_type='age',
                is_active=True
            ).update({ModelVersion.is_active: False})
            
            # Veritabanında yeni versiyon oluştur
            model_version = ModelVersion(
                model_type='age',
                version=new_version_num,
                version_name=version_name,
                created_at=datetime.now(),
                metrics=metadata['metrics'],
                is_active=True,
                training_samples=len(self.embedding_corrections),
                validation_samples=0,  # Ensemble için N/A
                epochs=0,  # Ensemble için N/A
                file_path=version_dir,
                weights_path=model_path,
                used_feedback_ids=self._get_used_feedback_ids()
            )
            
            db.session.add(model_version)
            db.session.commit()
            
            logger.info(f"✅ Ensemble model version created: {version_name}")
            logger.info(f"   Total corrections: {len(self.embedding_corrections)}")
            logger.info(f"   Embedding corrections: {len(self.embedding_corrections)}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error saving ensemble model version: {str(e)}")
            db.session.rollback()
            raise e
    
    def _get_correction_sources_stats(self) -> dict:
        """Get statistics about correction sources (embedding-based only)"""
        sources = {}
        
        # Sadece embedding corrections'dan kaynak istatistiklerini topla
        for correction in self.embedding_corrections.values():
            source = correction.get('source', 'UNKNOWN')
            sources[source] = sources.get(source, 0) + 1
        
        return sources
    
    def _calculate_ensemble_metrics(self) -> dict:
        """Ensemble metrikleri hesapla"""
        total_corrections = len(self.feedback_corrections)
        embedding_corrections = len(self.embedding_corrections)
        
        # Confidence ortalaması
        confidences = [c.get('confidence', 0.0) for c in self.feedback_corrections.values()]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_corrections': total_corrections,
            'embedding_corrections': embedding_corrections,
            'average_confidence': float(avg_confidence),
            'coverage_ratio': float(embedding_corrections / max(total_corrections, 1))
        }
    
    def _get_used_feedback_ids(self) -> list:
        """Get list of feedback IDs used in ensemble corrections (embedding-based only)"""
        feedback_ids = []
        
        # Sadece embedding corrections'dan feedback ID'leri topla
        for correction in self.embedding_corrections.values():
            if 'feedback_id' in correction:
                feedback_ids.append(correction['feedback_id'])
        
        return list(set(feedback_ids))  # Duplicate'leri çıkar 