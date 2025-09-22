from transformers import pipeline

# Load sentiment analysis model for predictions
sentiment_pipeline = pipeline("sentiment-analysis")


def predict_sentiment(text: str) -> dict:
    """
    Predicts sentiment by running sentiment analysis on the given text.
    Returns a dictionary with the label and confidence score.
    """
    result = sentiment_pipeline(text)[0]
    return {
        "label": result["label"],
        "score": float(result["score"])
    }


