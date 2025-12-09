from fastapi import FastAPI, File, UploadFile
from app.model import predict_genre
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    genre = predict_genre(file_path)
    
    os.remove(file_path)
    
    return {"genre": genre}