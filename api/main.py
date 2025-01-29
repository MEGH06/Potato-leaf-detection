from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model({ model path saved locally})
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I'm alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = np.array(image).astype(np.float32)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(float(np.max(predictions[0])) * 100, 2)
    return {
        "Predicted Class": predicted_class,
        "Confidence": f"{confidence}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
