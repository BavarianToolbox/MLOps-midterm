import numpy as np
import pickle
import pandas as pd

from PIL import Image
from io import BytesIO
from tensorflow import keras

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

input_shape = (32, 32)
labels = pd.read_csv('labels.csv', header = None)[0].to_list()

def unpickle(file: str):
    with open(file, 'rb') as fo:
        label_dict = pickle.load(fo, encoding='bytes')
    labels = [l.decode('utf-8') for l in label_dict[b'fine_label_names']]
    return labels


def read_image(img_encoded):
    '''Decode image and upload as PIL Image'''
    img_pil = Image.open(BytesIO(img_encoded))
    
    return img_pil


def preprocess(img: Image.Image):
    if img.size != input_shape:
        img = img.resize(input_shape)
    img = np.asarray(img, dtype = 'uint8')
    # normalize
    # img = img / 127.5 - 1.0
    # add batch dimension
    img = np.expand_dims(img, 0)

    return img


def load_model():
    model = keras.models.loadmodel('gs://constantin_midterm/train/models/model_00001/saved_model.pb')

    return model


def predict(img: np.ndarray):
    model = load_model()
    pred = model(img)
    pred_class = labels[np.argmax(pred)]

    return pred, pred_class
