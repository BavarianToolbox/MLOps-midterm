import wandb
import numpy as np
import pandas as pd

import tensorflow as tf

from PIL import Image
from io import BytesIO

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model = None
input_shape = (32, 32)
labels = pd.read_csv('labels.csv', header = None)[0].to_list()

def load_model():
    model = tf.keras.models.loadmodel('gs://constantin_midterm/train/models/model_00001/saved_model.pb')
    print('Model loaded')
    return model

_model = load_model()


def read_image(img_encoded):
    '''Decode image and upload as PIL Image'''
    img_pil = Image.open(BytesIO(img_encoded))
    
    return img_pil


def preprocess(img: Image.Image):
    if img.size != input_shape:
        img = img.resize(input_shape)
    # convert to numpy
    img = np.asarray(img, dtype = 'uint8')
    # add batch dimension
    img = np.expand_dims(img, 0)

    return img


def predict(img: np.ndarray):
    # load model
    global model
    if model is None:
        model = load_model()
    # predict
    img = preprocess(img)
    pred = model(img)
    pred = labels[np.argmax(pred)]

    return {'class': pred}
