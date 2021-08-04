from deploy.app.predict import preprocess, read_image
from typing import Optional
from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/api/predict")
def predict_image(file: UploadFile = File(...)):
    # decode and load image
    img = read_image(await file)
    # preprocess image
    img = preprocess(img)
    # predict
    