from fastapi import FastAPI
import uvicorn
from joblib import load
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from typing import List
from pydantic import BaseModel

class StringList(BaseModel):
    items: List[str]

nlp = spacy.load("it_core_news_sm")

svm_model = load('svm_model.joblib')

vectorizer = load('vectorizer.joblib')

stop_words = set(stopwords.words('italian'))

def preprocess_text(text):
    text = re.sub(r'(@[\w\d]+|#[\w\d]+|https://t\.co/[\w\d]+)', '', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text

def tokenizza_testo(testo):
    return word_tokenize(testo)

def lemmatizza_testo(parole_tokenizzate):
    parole_lemmatizzate = []
    for token in nlp(" ".join(parole_tokenizzate)):
        if token.lemma_.lower() not in stop_words:
            parole_lemmatizzate.append(token.lemma_)
    return parole_lemmatizzate

def rimuovi_stop_words(parole):
    return [parola for parola in parole if parola.lower() not in stop_words]



def processa_testo(testo):
    parole_tokenizzate = tokenizza_testo(testo)
    parole_pulite = rimuovi_stop_words(parole_tokenizzate)
    parole_lemmatizzate = lemmatizza_testo(parole_pulite)
    return parole_lemmatizzate


def predict_sentiment(text):
    processed_text = processa_testo(text)
    processed_text = ' '.join(processed_text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = svm_model.predict(text_vectorized)
    return prediction[0]

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict/{text}")
def predict(text: str):
    prediction = predict_sentiment(text)
    return {"text": text, "prediction": prediction}

@app.post("/predict")
def predict_post(comments: StringList):
    predictions = []
    for text in comments.items:
        prediction = predict_sentiment(text)
        predictions.append({"text": text, "prediction": prediction})
    predictions.sort(key=lambda x: x["prediction"], reverse=True)
    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)