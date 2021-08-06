# Deployment Details

The image classification application is containerized, monitored, and continuously deployed. More detailed information on each aspect of the application is listed below.

## Web Application

The web-app is built using FastAPI and deployed on a uvicorn server inside a container. The application uses a Tensorflow Keras EfficientNetB0 model fine-tuned on the CIFAR-100 dataset (refer to the [training repository](https://github.com/BavarianToolbox/MLOps-midterm/tree/main/train) for more details) to predict the class of any .jpg, .jpeg, or .png image. The trained model is loaded into the application from a GCP bucket when the `predict_image` POST method is invoked for the first time. Uploaded images are checked for a valid file extension and resized to the 32x32 size required by the model.

## Monitoring

The deployed web-app is monitored using Weights and Biases (W&B). In addition to the standard system metrics tracked by W&B (Network Traffic, Dick Utilization, CPU Threads In Use, etc.) detailed prediction information from each inference is also tracked. 

The Prediction Information Table tracks input image statistics (original width, height, and a copy of the preprocessed input image) and prediction information (the index and value of the maximum prediction from the final 100-dimensional, sigmoid-activated output layer, and the predicted class from the CIFAR-100 dataset).

![Prediction Information Table](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/deploy/figures/Prediction_Information_Table.png "Prediction Information Table")

The Raw Prediction Table tracks the predicted class, determined based on the index of the maximum value from the final 100-dimensional output layer, and all the values from the final 100-dimensional layer.

![Raw Prediction Table](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/deploy/figures/Raw_Prediction_Table.png "Raw Prediction Table")

The W&B credentials necessary to log information to a specific W&B account are stored in the Google Secret Manager and loaded as an environment variable when the container is deployed. 

## Containerization

The application is containerized with Google Cloud Build according to the specifications in the [Dockerfile](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/deploy/Dockerfile). Upon successfully building the container it is pushed to the Google Container Registry and deployed using Google Cloud Run.

## Continuous Deployment