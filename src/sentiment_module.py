from transformers import pipeline

_sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,
)

def analyze_sentiment(text: str):
    result = _sentiment_pipe(text[:512])[0]
    return {"label": result["label"], "score": float(result["score"])}
