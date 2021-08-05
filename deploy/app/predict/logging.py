import wandb
import os
import datetime
from google.cloud import storage


def gcp_init():
    '''Initialize GCP Bucket connection'''
    # gcp bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket('constantin_midterm')


def wandb_init():
    '''Initialize W&B run'''
    # wandb key

    # cloud version
    blob = bucket.blob('train/keys/wandb_key.json')
    wandb_key = json.loads(blob.download_as_string())

    # local version
    # os.environ["WANDB_API_KEY"] = "..."

    # global run 
    run = wandb.init(
        project = 'midterm-prod-monitor-dev',
        group = wandb.util.generate_id(),
        tags = ["Production"],
        config = {'launch':str(datetime.datetime.now())}
    )


def wandb_end():
    wandb.finish()