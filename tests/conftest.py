"""
Pytest configuration for WSANALIZ tests
"""
import pytest
import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def app():
    """Create application for testing"""
    from app import create_app
    
    created = create_app('testing')
    # create_app artık (app, socketio) tuple döndürüyor olabilir
    app = created[0] if isinstance(created, tuple) else created
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False
    })
    
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create CLI runner"""
    return app.test_cli_runner() 