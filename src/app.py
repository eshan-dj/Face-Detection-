from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import mlflow.pyfunc

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/FaceDetection2/Production")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).resize((150,150))
    img_array = np.array(img) / 255.0
    pred = model.predict(np.expand_dims(img_array, axis=0))
    return {"class": int(np.argmax(pred)), "confidence": float(np.max(pred))}