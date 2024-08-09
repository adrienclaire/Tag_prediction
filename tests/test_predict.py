# tests/test_predict.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    response = client.post('/predict', json={"text": "This is a test question"})
    assert response.status_code == 200

def test_predict_content(client):
    response = client.post('/predict', json={"text": "How to install Python packages?"})
    data = response.get_json()
    assert 'predicted_tags' in data
    assert isinstance(data['predicted_tags'], list)