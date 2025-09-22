from fastapi.testclient import TestClient
from app.main import app, predict_text_sentiment
from app.schemas import SentimentRequest

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Sentiment Analysis API"}


def test_predict_text_sentiment_positive():
    request = SentimentRequest(text="I love this!")
    result = predict_text_sentiment(request)
    response_body = result.model_dump()
    assert response_body["label"] == "POSITIVE"
    assert 0 <= response_body["score"] <= 1


def test_predict_text_sentiment_negative():
    request = SentimentRequest(text="I hate this!")
    result = predict_text_sentiment(request)
    response_body = result.model_dump()
    assert response_body["label"] == "NEGATIVE"
    assert 0 <= response_body["score"] <= 1


# direct API integration test
def test_predict_text_sentiment_api():
    response = client.post("/predict", json={"text": "I'm having a great day!"})
    assert response.status_code == 200
    response_body = response.json()
    assert "label" in response_body
    assert "score" in response_body
