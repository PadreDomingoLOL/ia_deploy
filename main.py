import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

port = 8000

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/predict/")
async def predict(file: UploadFile = File(...)):
    logging.info("api predict called")
    try:
        logging.info("Loading model")
        model = load_model("model1.h5")
        logging.info("Model loaded successfully")
        
        # Leer la imagen recibida
        contents = await file.read()
        logging.info("File read successfully")
        img = Image.open(io.BytesIO(contents))
        
        # Preprocesar la imagen
        img = img.resize((224, 224))
        img = img.convert('RGB')
        image_array = img_to_array(img)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        logging.info("Image preprocessed successfully")
        
        # Hacer la predicción
        prediction = model.predict(image_array)
        logging.info("Prediction made successfully")
        result = "Severo" if prediction[0][0] > 0.5 else "No Severo"
        
        return {"prediction": result}
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}