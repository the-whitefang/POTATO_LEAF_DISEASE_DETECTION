import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

template = Jinja2Templates(directory="templates")  # register template engine

app = FastAPI()
# CORS middleware configuration
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # This allows all headers
)
MODEL = tf.keras.models.load_model('C:/Users/abhil/potato_disease/models/1')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return template.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predictpage(request: Request):
    return template.TemplateResponse("predict.html", {"request": request, "cls": "-", "clscnf": "-", "status": "none"})

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    # image = read_file_as_image("C:/Users/abhil/potato_disease/training/PlantVillage/Potato___healthy/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG")
    # print(image)
    # return {"filename": file.filename}
    image_batch = np.expand_dims(image, 0)
    #
    prediction = MODEL.predict(image_batch)
    # print(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.argmax(prediction[0])

    # return {
    #     'class': predicted_class,
    #     'confidence': float(confidence)
    # }
    return template.TemplateResponse("predict.html", {"request": request, "cls": predicted_class, "clscnf": float(confidence), "status": "block"})


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
