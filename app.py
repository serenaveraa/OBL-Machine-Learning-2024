from fastapi import FastAPI
from joblib import load
import pandas as pd

app = FastAPI()
model = load('tree.joblib')

@app.get("/predictions")
async def prediction(Year: int, sepal: int, kdkd: int):
    data = pd.DataFrame([Year, sepal, kdkd], 
        columns=['Year', 'sepal', 'kdkd']) 
    prediction = model.predict(data)
    return {'prediction': int(prediction[0])}


#uvicorn app:app --reload