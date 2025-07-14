import os
import json
import shutil
from datetime import datetime
from flask import current_app
import logging
from config import Config
from app.utils.file_utils import ensure_dir, safe_copytree, safe_remove, write_json, read_json, get_folder_size

class CLIPVersionService:
    """CLIP Model Version Management Service"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clip_models_path = os.path.join(current_app.config['MODELS_FOLDER'], 'clip')
        self.base_model_path = os.path.join(self.clip_models_path, 'ViT-H-14-378-quickgelu_dfn5b', 'base_model')
        self.active_model_path = os.path.join(self.clip_models_path, 'ViT-H-14-378-quickgelu_dfn5b', 'active_model')
        self.versions_path = os.path.join(self.clip_models_path, 'versions')
        
        # Versions klasörünün var olduğundan emin ol
        ensure_dir(self.versions_path)
    
    def get_all_versions(self):
        """Tüm CLIP versiyonlarını listele"""
        try:
            versions = []
            
            # Base model bilgisi
            if os.path.exists(self.base_model_path):
                base_info = self._get_model_info(self.base_model_path, 'base')
                versions.append(base_info)
            
            # Active model bilgisi
            if os.path.exists(self.active_model_path):
                active_info = self._get_model_info(self.active_model_path, 'active')
                versions.append(active_info)
            
            # Version klasöründeki modeller
            if os.path.exists(self.versions_path):
                for item in os.listdir(self.versions_path):
                    item_path = os.path.join(self.versions_path, item)
                    
                    # JSON session dosyaları
                    if item.endswith('.json'):
                        session_info = self._get_session_info(item_path)
                        if session_info:
                            versions.append(session_info)
                    
                    # Model klasörleri
                    elif os.path.isdir(item_path):
                        version_info = self._get_model_info(item_path, 'version')
                        versions.append(version_info)
            
            # Tarihe göre sırala (en yeni önce)
            versions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return versions, None
            
        except Exception as e:
            self.logger.error(f"Versiyonlar listelenirken hata: {str(e)}")
            return [], str(e)
    
    def _get_model_info(self, model_path, model_type):
        """Model bilgilerini getir"""
        try:
            info = {
                'path': model_path,
                'type': model_type,
                'name': os.path.basename(model_path),
                'size': self._get_folder_size(model_path),
                'created_at': datetime.fromtimestamp(os.path.getctime(model_path)).isoformat(),
                'is_active': model_type == 'active'
            }
            
            # Metadata dosyası varsa oku
            metadata_file = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    info.update(metadata)
            
            return info
            
        except Exception as e:
            self.logger.error(f"Model bilgisi alınırken hata: {str(e)}")
            return {
                'path': model_path,
                'type': model_type,
                'name': os.path.basename(model_path),
                'error': str(e)
            }
    
    def _get_session_info(self, session_file):
        """Training session bilgilerini getir"""
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            session_data['type'] = 'training_session'
            session_data['file_path'] = session_file
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"Session bilgisi alınırken hata: {str(e)}")
            return None
    
    def _get_folder_size(self, folder_path):
        """Klasör boyutunu hesapla (MB)"""
        return get_folder_size(folder_path)
    
    def create_new_version(self, version_name, source_model='base'):
        """Yeni model versiyonu oluştur"""
        try:
            # Source model path
            if source_model == 'base':
                source_path = self.base_model_path
            elif source_model == 'active':
                source_path = self.active_model_path
            else:
                return None, f"Geçersiz source model: {source_model}"
            
            if not os.path.exists(source_path):
                return None, f"Source model bulunamadı: {source_path}"
            
            # Yeni versiyon path
            new_version_path = os.path.join(self.versions_path, version_name)
            
            if os.path.exists(new_version_path):
                return None, f"Versiyon zaten mevcut: {version_name}"
            
            # Model'i kopyala
            self.logger.info(f"Model kopyalanıyor: {source_path} -> {new_version_path}")
            safe_copytree(source_path, new_version_path)
            
            # Metadata oluştur
            metadata = {
                'version_name': version_name,
                'source_model': source_model,
                'created_at': datetime.now().isoformat(),
                'created_by': 'clip_training_service',
                'model_type': 'fine_tuned_clip',
                'status': 'created'
            }
            
            metadata_file = os.path.join(new_version_path, 'metadata.json')
            write_json(metadata_file, metadata)
            
            self.logger.info(f"Yeni versiyon oluşturuldu: {version_name}")
            return new_version_path, None
            
        except Exception as e:
            self.logger.error(f"Yeni versiyon oluşturulurken hata: {str(e)}")
            return None, str(e)
    
    def set_active_model(self, version_path):
        """Belirtilen versiyonu aktif model yap"""
        try:
            if not os.path.exists(version_path):
                return False, f"Versiyon bulunamadı: {version_path}"
            
            # Mevcut active model'i backup'la
            backup_path = os.path.join(self.versions_path, f"backup_active_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if os.path.exists(self.active_model_path):
                self.logger.info(f"Mevcut active model backup'lanıyor: {backup_path}")
                safe_copytree(self.active_model_path, backup_path)
                safe_remove(self.active_model_path)
            
            # Yeni versiyonu active yap
            self.logger.info(f"Yeni active model ayarlanıyor: {version_path}")
            safe_copytree(version_path, self.active_model_path)
            
            # Active model metadata'sını güncelle
            metadata = {
                'activated_at': datetime.now().isoformat(),
                'source_version': os.path.basename(version_path),
                'backup_path': backup_path,
                'model_type': 'active_clip'
            }
            
            metadata_file = os.path.join(self.active_model_path, 'metadata.json')
            write_json(metadata_file, metadata)
            
            self.logger.info(f"Active model güncellendi: {os.path.basename(version_path)}")
            return True, None
            
        except Exception as e:
            self.logger.error(f"Active model ayarlanırken hata: {str(e)}")
            return False, str(e)
    
    def delete_version(self, version_name):
        """Versiyon sil"""
        try:
            version_path = os.path.join(self.versions_path, version_name)
            
            if not os.path.exists(version_path):
                return False, f"Versiyon bulunamadı: {version_name}"
            
            # Base ve active model'leri silmeye izin verme
            if version_name in ['base_model', 'active_model']:
                return False, "Base ve active model silinemez"
            
            # Versiyonu sil
            safe_remove(version_path)
            
            self.logger.info(f"Versiyon silindi: {version_name}")
            return True, None
            
        except Exception as e:
            self.logger.error(f"Versiyon silinirken hata: {str(e)}")
            return False, str(e)
    
    def compare_versions(self, version1_path, version2_path):
        """İki versiyonu karşılaştır"""
        try:
            comparison = {
                'version1': self._get_model_info(version1_path, 'comparison'),
                'version2': self._get_model_info(version2_path, 'comparison'),
                'size_difference': 0,
                'metadata_diff': {}
            }
            
            # Boyut farkı
            size1 = comparison['version1'].get('size', 0)
            size2 = comparison['version2'].get('size', 0)
            comparison['size_difference'] = size2 - size1
            
            return comparison, None
            
        except Exception as e:
            self.logger.error(f"Versiyonlar karşılaştırılırken hata: {str(e)}")
            return None, str(e)
    
    def get_active_model_info(self):
        """Aktif model bilgilerini getir"""
        try:
            if not os.path.exists(self.active_model_path):
                return None, "Active model bulunamadı"
            
            info = self._get_model_info(self.active_model_path, 'active')
            return info, None
            
        except Exception as e:
            self.logger.error(f"Active model bilgisi alınırken hata: {str(e)}")
            return None, str(e) 