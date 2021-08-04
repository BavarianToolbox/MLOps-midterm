from deploy.app.predict import predict, read_image
from typing import Optional
from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    # decode and load image
    img = read_image(await file)
    # predict
    pred = predict(img)
    print(f'Predicted class: {pred}')
    return pred