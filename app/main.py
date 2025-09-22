from fastapi import FastAPI
from app.model import predict_sentiment
from app.schemas import SentimentRequest, SentimentResponse

app = FastAPI(title="Sentiment Analysis API")


@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API"}


@app.post("/predict", response_model=SentimentResponse)
def predict_text_sentiment(request: SentimentRequest):
    """
    Text Sentiment prediction.
    :param request: Takes in a text string sentence.
    :return: response: Returns a sentiment prediction.
    """
    result = predict_sentiment(request.text)
    return SentimentResponse(**result)
