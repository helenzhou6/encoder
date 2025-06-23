import wandb

def get_device():
    return "cpu"

def init_wandb(config={}):
    default_config = {
        "learning_rate": 0.1,
        "architecture": "Encoder",
        "dataset": "MNIST",
        "epochs": 5,
    }
    # Start a new wandb run to track this script.
    return wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="week3-rebels",
        # Set the wandb project where this run will be logged.
        project="DigitFinder",
        # Track hyperparameters and run metadata.
        config={**default_config, **config},
    )
