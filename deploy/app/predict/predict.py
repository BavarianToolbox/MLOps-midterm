import wandb
import numpy as np
import pandas as pd
import matplotlib.cm as cm

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
    print('Loading model')
    # GCP version
    model = tf.keras.models.load_model('gs://constantin_midterm/train/models/model_00002')
    # local version
    # model = tf.keras.models.load_model('/app/model_00002')
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


def superimpose(img, heatmap, alpha=0.4):
    '''Based on: https://keras.io/examples/vision/grad_cam/'''
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


def get_gradcam_image(img_array, model, last_conv_layer_name, pred_index=None):
    '''Based on: https://keras.io/examples/vision/grad_cam/'''
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # superimpose heatmap onto input image
    superimposed_img = superimpose(np.squeeze(img_array), heatmap)

    return superimposed_img


def predict(img: np.ndarray):
    # load model
    global model
    if model is None:
        model = load_model()
    # predict
    img, orig_img_size = preprocess(img)
    preds_raw = model(img)
    preds = preds_raw[0].numpy().tolist()
    max_pred_idx = int(np.argmax(preds))
    max_pred = preds[max_pred_idx]
    pred_class = labels[max_pred_idx]
    
    # get gradcam image
    # This is pointless with the EfficientNetB0 model when using input
    # images of size 32x32 because the activations from the final convolution
    # are 1x1x1280. Grad-CAM requires expanding the activations form the final convolution
    # to the size of the input image. Expanding a 1x1 tensor to 32x32 would result in all
    # the same values throughout the 32x32 heatmap, which is pointless.
    
    # gradcam_image = get_gradcam_image(img, model, 'top_conv')

    # log to w&b
    cols_1 = [
        'Original image width',
        'Original image height',
        'Index of max prediction',
        'Max prediction',
        'Predicted class',
        'Input image'
        #'Input image with Grad-CAM heatmap'
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
        # wandb.Image(
        #     gradcam_image,
        #     caption = 'Grad-CAM heatmap'
        # )
    ]]

    cols_2 = ['Predicted class', *labels]
    data_2 = [pred_class, *preds]

    wandb.run.log({
        'Prediction Information Table' : wandb.Table(
            columns=cols_1,
            data = data_1,
            allow_mixed_types=True
        ),
        'Raw Prediction Table': wandb.Table(
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