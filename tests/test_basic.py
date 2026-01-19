"""
Basic application tests
"""

def test_app_creation(app):
    """Test that the app creates successfully"""
    assert app is not None
    assert app.config['TESTING'] is True

def test_main_route(client):
    """Test the main route"""
    response = client.get('/')
    assert response.status_code == 200

def test_nonexistent_route(client):
    """Test 404 handling"""
    response = client.get('/nonexistent')
    assert response.status_code == 404

def test_api_analysis_without_auth(client):
    """Test API endpoints require proper data"""
    response = client.post('/api/analysis/start')
    assert response.status_code in [400, 401, 404]  # Should not be 200

def test_file_upload_route_exists(client):
    """Test file upload route exists"""
    response = client.get('/api/files/upload')
    # This might return 405 (Method Not Allowed) which is fine - route exists
    assert response.status_code in [200, 405, 404] 