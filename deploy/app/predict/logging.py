import wandb
import datetime


def wandb_init():
    '''Initialize W&B run'''

    # local version
    # os.environ["WANDB_API_KEY"] = "..."
    # cloud version sets ENV variable with Secret manager and Cloud Run

    # global run 
    run = wandb.init(
        project = 'midterm-prod-monitor-gcp',
        group = wandb.util.generate_id(),
        tags = ["Production", "GCP"],
        config = {'launch':str(datetime.datetime.now())}
    )


def wandb_end():
    wandb.finish()