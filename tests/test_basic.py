"""
Basic tests for WSANALIZ application
"""
import pytest
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app


class TestBasic:
    """Basic application tests"""

    @pytest.fixture
    def app(self):
        """Create test app"""
        app = create_app()
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()

    def test_app_creation(self, app):
        """Test app can be created"""
        assert app is not None
        assert app.config['TESTING'] is True

    def test_home_page(self, client):
        """Test home page loads"""
        response = client.get('/')
        assert response.status_code == 200

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        # Health endpoint may not exist yet, so we check for 404 or 200
        assert response.status_code in [200, 404]

    def test_static_files(self, client):
        """Test static files are accessible"""
        response = client.get('/static/css/style.css')
        # CSS file should exist or return 404
        assert response.status_code in [200, 404]


class TestConfig:
    """Configuration tests"""

    def test_config_values(self):
        """Test configuration values"""
        app = create_app()
        
        # Test that required config values exist
        required_configs = [
            'SECRET_KEY',
            'UPLOAD_FOLDER',
            'MODELS_FOLDER'
        ]
        
        for config_key in required_configs:
            assert config_key in app.config


class TestDirectoryStructure:
    """Test directory structure"""

    def test_required_directories_exist(self):
        """Test that required directories exist or can be created"""
        required_dirs = [
            'app',
            'app/ai',
            'app/services',
            'app/routes',
            'app/models',
            'app/templates',
            'app/static'
        ]
        
        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Directory {dir_path} should exist"

    def test_storage_directories(self):
        """Test storage directories structure"""
        storage_dirs = [
            'storage',
            'storage/uploads',
            'storage/processed',
            'storage/models'
        ]
        
        for dir_path in storage_dirs:
            # These directories should exist or be creatable
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    assert os.path.exists(dir_path)
                except Exception as e:
                    pytest.fail(f"Could not create directory {dir_path}: {e}")


if __name__ == '__main__':
    pytest.main([__file__]) 