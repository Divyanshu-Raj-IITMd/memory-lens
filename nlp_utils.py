# nlp_utils.py

from transformers import pipeline

# Load an emotion detection pipeline
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

# Load a summarization pipeline
summary_model = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def get_emotion(text: str) -> str:
    """
    Detect the most probable emotion from the given text.
    Returns a string like "joy", "sadness", etc.
    """
    result = emotion_model(text)[0]
    return result['label']

def get_summary(text: str) -> str:
    """
    Summarize the given text.
    Uses a BART model to produce a short summary.
    """
    summary = summary_model(
        text,
        max_length=50,
        min_length=10,
        do_sample=False
    )
    return summary[0]['summary_text']
