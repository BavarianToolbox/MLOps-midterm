# Training Details

## Model

## Monitoring

The training process is monitored using Weights and Biases (W&B). In addition to the standard system metrics tracked by W&B (Network Traffic, Dick Utilization, CPU Threads In Use, etc.) detailed prediction information from each inference is also tracked. 

`train.py` contains the code for training the model, monitoring the training with W&B, and saving the model. The code is designed to be executed in a GCP VM so that data, credentials, and models can be read-from and written-to GCP Buckets without the need for explicit GCP credentials. The `nbs` folder contains jupyter notebooks used to aid in the development process, such as exploratory data analysis [EDA.ipynb](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/train/nbs/EDA.ipynb) and a training playground [training_playground.ipynb](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/train/nbs/training_playground.ipynb).