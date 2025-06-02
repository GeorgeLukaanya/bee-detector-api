# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2
import numpy as np
import os
from io import BytesIO

app = FastAPI()

# Load YOLO model
model = YOLO("bee-detector.pt")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    results = model(image)

    detections = results[0].tojson()
    return JSONResponse(content={"detections": detections})

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    results = model(tmp_path)
    detections = results[0].tojson()
    os.remove(tmp_path)

    return JSONResponse(content={"detections": detections})
