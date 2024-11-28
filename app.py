from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd

# Inicializar la aplicación FastAPI
app = FastAPI()

# Cargar el modelo previamente entrenado
model = load("best_gradient_boosting_model.joblib")

# Modelo de entrada para predicción (todos los datos como float)
class PredictionInput(BaseModel):
    Year: float
    Global: float
    Platform: float
    Genre: float
    Game_Age: float
    NorthAmerica_Global_Ratio: float
    Europe_Global_Ratio: float
    Japan_Global_Ratio: float
    RestOfWorld_Global_Ratio: float

# Endpoint raíz
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de prediccion con Gradient Boosting"}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        # Convertir los datos de entrada a un DataFrame de pandas
        input_df = pd.DataFrame([input_data.dict()])

        # Validar dimensiones de entrada
        if input_df.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"El modelo espera {model.n_features_in_} características, pero se proporcionaron {input_df.shape[1]}"
            )

        # Asegurar que los datos son `float64`
        input_df = input_df.astype("float64")

        # Hacer predicción
        prediction = model.predict(input_df)

        # Devolver el resultado
        return {"prediction": prediction.tolist()}

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail="Error durante la predicción. Verifica los datos de entrada.")