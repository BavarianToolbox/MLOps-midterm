import wandb
import numpy as np
import pandas as pd
import os

import tensorflow as tf
# from google.cloud import storage

from PIL import Image
from io import BytesIO

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model = None
input_shape = (32, 32)
labels = pd.read_csv('/app/predict/labels.csv', header = None)[0].to_list()





def load_model():
    '''Load Keras model from GCP Bucket'''
    # GCP version
    model = tf.keras.models.loadmodel('gs://constantin_midterm/train/models/model_00001')
    # local version
    print('Loading model')
    # model = tf.keras.models.load_model('/app/model_00001')
    print('Model loaded')

    return model


def read_image(img_encoded):
    '''Decode image and upload as PIL Image'''
    img_pil = Image.open(BytesIO(img_encoded))
    
    return img_pil


def preprocess(img: Image.Image):
    '''Preprocess image for model prediction'''
    orig_size = img.size
    if orig_size != input_shape:
        img = img.resize(input_shape)
    img = np.asarray(img, dtype = 'uint8')
    img = np.expand_dims(img, 0)

    return img, orig_size


def predict(img: np.ndarray):
    # load model
    global model
    if model is None:
        model = load_model()
    # predict
    img, orig_img_size = preprocess(img)
    preds = model(img)[0].numpy().tolist()
    max_pred_idx = int(np.argmax(preds))
    max_pred = preds[max_pred_idx]
    pred_class = labels[max_pred_idx]

    # log to w&b
    cols_1 = [
        'Original image width',
        'Original image height',
        'Index of max prediction',
        'Max prediction',
        'Predicted class',
        'Input image'
    ]
    data_1 = [[
        orig_img_size[0],
        orig_img_size[1],
        max_pred_idx,
        max_pred,
        pred_class,
        wandb.Image(
            Image.fromarray(np.squeeze(img)), 
            caption = f'Predicted class: {pred_class}'
        )
    ]]
    cols_2 = ['Predicted class', *labels]
    data_2 = [pred_class, *preds]
    wandb.run.log({
        'Prediction Information' : wandb.Table(
            columns=cols_1,
            data = data_1,
            allow_mixed_types=True
        ),
        'Raw Prediction': wandb.Table(
            columns = cols_2,
            data = [data_2],
            allow_mixed_types=True
        )
    })

    return {
        'max_pred_idx': max_pred_idx,
        'max_pred': max_pred,
        'class': pred_class
    }