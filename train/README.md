# Training Details

## Data

The model is trained on the CIFAR-100 dataset. The [EDA Notebook](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/train/nbs/EDA.ipynb) contains statistics about the dataset, including the number of classes, the number of samples, and the distribution of the classes in the training and validation partitions.

## Model

The model architecture is based on the EfficientNetB0 model with a final layer that has 100 output nodes corresponding to the 100 classes in the CIFAR-100 dataset. The model is fine-tuned (only the final fully connected layer is trained, the rest of the model uses pre-trained ImageNet weights).

## Training

The `train.py` file contains the code for training the model, monitoring the training, and saving the model to a GCP bucket. The srcipt requires two arguments: 

- `project`: the name of the Weights & Biases project the training metrics are logged to. 
- `cfg`: a configuration JSON file specifying the training parameters

The code is designed to be executed in a GCP VM so that data, credentials, and models can be read-from and written-to the GCP Bucket without the need for explicit GCP credentials. The W&B project name and configuration file are passed to the `Docker run` command. The file, loaded from a GCP bucket, contains detailed training parameters, such as the number of epochs, batch-size, learning rate, and optimizer. Training new versions of the model is as easy as creating a new configuration JSON file, uploading it to the bucket, and spinning up another container. The final model is automatically saved in a GCP bucket according to a location specified in the config file.

## Monitoring

The training process is monitored using Weights and Biases (W&B). In addition to the standard system metrics tracked by W&B (Network Traffic, Dick Utilization, CPU Threads In Use, etc.) the loss and accuracy for the training and validation data are tracked for each epoch (see graphs below)

![W&B Training Monitor Graphs](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/train/figures/W%26B_Training_Graphs.png "W&B Training Monitor Graphs")

 The `nbs` folder contains jupyter notebooks used to aid in the development process, such as exploratory data analysis [EDA.ipynb](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/train/nbs/EDA.ipynb) and a training playground [training_playground.ipynb](https://github.com/BavarianToolbox/MLOps-midterm/blob/main/train/nbs/training_playground.ipynb).