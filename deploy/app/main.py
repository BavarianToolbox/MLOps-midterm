from .predict.predict import predict, read_image
from .predict.logging import gcp_init, wandb_init, wandb_end
from typing import Optional
from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI()


@app.on_event('startup')
async def startup_event():
    # print('Initializing GCP connection.')
    # gcp_init()
    print('Initializing W&B run.')
    wandb_init()


@app.on_event('shutdown')
async def shutdown_event():
    wandb_end()
    print('W&B run ended.')


@app.get("/")
def read_root():
    
    return {"Hello": "World"}


@app.post("/predict/cifar100")
async def predict_image(file: UploadFile = File(...)):
    # check file extension
    extension = file.filename.split('.')[-1]
    if extension not in ('jpg', 'jpeg', 'png'):
        return f'{file.filename} is not the propper format! \
            Must be .jpg, .jpeg, or .png'
    # decode and load image
    img = read_image(await file.read())
    # predict
    pred = predict(img)
    print(
        f"Index of maximum prediction: {pred['max_pred_idx']}\n \
        Maximum predicted value: {pred['max_pred']}\n \
        Predicted class: {pred['class']}"
    )
    
    return pred