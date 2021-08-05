# import wandb
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


# # gcp bucket
# storage_client = storage.Client()
# bucket = storage_client.bucket('constantin_midterm')

# # wandb key
# blob = bucket.blob('train/keys/wandb_key.json')
# wandb_key = json.loads(blob.download_as_string())
# os.environ["WANDB_API_KEY"] = wandb_key['key']

# run = wandb.init(project='midterm-prod-monitor')


def load_model():
    # GCP version
    # model = tf.keras.models.loadmodel('gs://constantin_midterm/train/models/model_00001')
    # local version
    model = tf.keras.models.load_model('/app/model_00001')
    print('Model loaded')
    return model


def read_image(img_encoded):
    '''Decode image and upload as PIL Image'''
    img_pil = Image.open(BytesIO(img_encoded))
    
    return img_pil


def preprocess(img: Image.Image):
    orig_size = img.size
    if orig_size != input_shape:
        img = img.resize(input_shape)
    # convert to numpy
    img = np.asarray(img, dtype = 'uint8')
    # add batch dimension
    img = np.expand_dims(img, 0)

    return img, orig_size


def predict(img: np.ndarray):
    # load model
    global model
    if model is None:
        model = load_model()
    # predict
    img, orig_img_size = preprocess(img)
    pred = model(img)
    pred_class = labels[np.argmax(pred)]
    # wandb.log({
    #     'original_image_size': orig_img_size,
    #     'image': wandb.Image(img),
    #     'prediction': pred,
    #     'predicted_class':pred_class
    # })
    return {'class': pred_class}
