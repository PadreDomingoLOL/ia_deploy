from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

origins = [
    "*",
    "http://localhost",
    "http://localhost:3000",
    "https://hamec.vercel.app/"
]

app = FastAPI(title="ECG Classification API", description="API para clasificar ECGs", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

port = 8000

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/predict/")
async def predict(file: UploadFile = File(...)):
    print("api predict")
    try:
        model = load_model("model1.h5")
        print("modelo cargado")
        # Leer la imagen recibida
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocesar la imagen
        img = img.resize((224, 224))
        img = img.convert('RGB')
        image_array = img_to_array(img)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Hacer la predicción
        prediction = model.predict(image_array)
        result = "Severo" if prediction[0][0] > 0.5 else "No Severo"
        
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}