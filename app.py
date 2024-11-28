from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import json

app = FastAPI()

# Cargar el modelo entrenado
model = load_model("game_genre_model.h5")

from tensorflow.keras.preprocessing.text import tokenizer_from_json

with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

# Cargar el codificador de etiquetas
label_encoder = load("label_encoder.joblib")

# Parámetros de preprocesamiento
max_length = 50  # Longitud máxima usada al entrenar

# Mapeo de géneros (LabelEncoder invertido)
genre_map = {0: "Action", 1: "Adventure", 2: "RPG", 3: "Simulation", 4: "Sports"}

# Clase para recibir datos en el cuerpo de la solicitud
class GameSummary(BaseModel):
    summary: str

@app.post("/predict-genre")
async def predict_genre(game: GameSummary):
    # Preprocesar el texto recibido
    text = game.summary
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Realizar la predicción
    predictions = model.predict(padded_sequence)
    predicted_genre_idx = np.argmax(predictions[0])  # Índice con mayor probabilidad
    predicted_genre = genre_map[predicted_genre_idx]

    return {
        "summary": text,
        "predicted_genre": predicted_genre,
        "confidence": float(predictions[0][predicted_genre_idx])
    }
