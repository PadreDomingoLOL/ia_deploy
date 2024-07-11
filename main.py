from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import requests
import os
from dotenv import load_dotenv

load_dotenv()

origins = [
    "*",
    "http://localhost",
    "http://localhost:3000",
    "https://hamec.vercel.app/"
]

modelId = os.getenv("MODEL_ID")

url = f"https://drive.google.com/uc?id={modelId}"

downloaded = False

app = FastAPI(title="ECG Classification API", description="API para clasificar ECGs", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def download_model():
    global downloaded
    print("Descargando modelo")
    try:
        response = requests.get(url)
        with open("model1.h5", 'wb') as f:
            f.write(response.content)
        downloaded = True
        return {"message": "Modelo descargado"}
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    global downloaded 
    downloaded = True
    await download_model()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/predict/")
async def predict(file: UploadFile = File(...)):
    global downloaded
    if (not UploadFile):
        return {'File not uploaded'}
    if (not downloaded):
        print("Modelo no disponible...")
        await download_model()
        print("Modelo descargado")
    else:
        return {'MOdelo pasado mal'}
    print("Modelo ecnontrado")
    try:
        model = load_model("model1.h5")
        print("modelo cargado")
        # Leer la imagen recibida
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocesar la imagen
        img = img.resize((224, 224))
        image_array = img_to_array(img)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Hacer la predicción
        prediction = model.predict(image_array)
        result = "Severo" if prediction[0][0] > 0.5 else "No Severo"
        
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}