from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Sentiment Analysis API"}


def test_predict_text_sentiment():
    response = client.post("/predict", json={"text": "I'm having a great day!"})
    assert response.status_code == 200
    response_body = response.json()
    assert "label" in response_body
    assert "score" in response_body
