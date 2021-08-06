import wandb
import os
import json
import datetime
from google.cloud import storage


def wandb_init():
    '''Initialize W&B run'''

    # local version
    os.environ["WANDB_API_KEY"] = "..."

    # global run 
    run = wandb.init(
        project = 'midterm-prod-monitor-dev-gcp',
        group = wandb.util.generate_id(),
        tags = ["Production"],
        config = {'launch':str(datetime.datetime.now())}
    )


def wandb_end():
    wandb.finish()