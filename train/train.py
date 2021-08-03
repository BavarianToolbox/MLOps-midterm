import pickle
import json
import argparse
import os
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from google.cloud import storage

import wandb
from wandb.keras import WandbCallback

# gcp bucket
storage_client = storage.Client()
bucket = storage_client.bucket('constantin_midterm')

# wandb key
blob = bucket.blob('train/keys/wandb_key.json')
wandb_key = json.loads(blob.download_as_string())
os.environ["WANDB_API_KEY"] = wandb_key['key']


def load_data(file_name: str):
    '''Load data from bucket'''
    print(f'Loading data: {file_name}')

    # load from bucket
    blob = bucket.blob(f'train/data/{file_name}')
    [x_train, y_train, x_test, y_test] = pickle.loads(blob.download_as_string())

    # check shape
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    return x_train, y_train, x_test, y_test


def get_model(config):
    '''Build and compile model'''
    print('Building model')
    # pretrained model base
    base = keras.applications.EfficientNetB0(
        include_top = False,
        weights = 'imagenet',
        input_shape = [32, 32, 3],
        pooling = config.pool
    )

    # fine-tune or full-train
    base.trainable = not bool(config.fine_tune)
    
    # assemble model
    model = keras.Sequential([
        base,
        keras.layers.Dense(100, activation = 'sigmoid')
    ])
    print('model.summary()')
    
    # set optimizer and compile
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    model.compile(
        optimizer = optimizer,
        loss = config.loss,
        metrics = config.metrics
    )
    
    return model


def load_run_config(file_name: str):
    blob = bucket.blob(f'train/config/{file_name}')
    config = json.loads(blob.download_as_string())
    
    return config


def train(project: str, config: dict):
    '''W&B monitor initialization, training, and model saving'''
    
    # set up W&B run
    run = wandb.init(project=project, config=config)
    config = wandb.config
    
    # load data and model
    x_train, y_train, x_test, y_test = load_data(config.data_file)
    model = get_model(config)
    
    # train
    model.fit(
        x = x_train,
        y = y_train,
        batch_size = config.batch_size,
        validation_data = (x_test, y_test),
        epochs = config.epochs,
        callbacks = [WandbCallback()]
    )

    # save model and finish W&B run
    print(f'Saving model under: {config.model_version}')
    model.save(f'gs://constantin_midterm/train/models/{config.model_version}')
    run.finish()


def get_args():
    parser = argparse.ArgumentParser(description='Specify training configuration file.')
    parser.add_argument('--project', '-p', type=str, default='mlops-midterm',
        help='project name for W&B monitoring', required=True)
    parser.add_argument('--cfg', '-c', type=str, 
        help='a json file specifying training configurations', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # print(args.project)
    # print(args.cfg)
    proj = args.project
    config = load_run_config(args.cfg)
    train(proj, config)
    quit()