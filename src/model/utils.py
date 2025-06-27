import wandb
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_artifact(model_name, model_description, file_extension='pt', type="model"):
    artifact = wandb.Artifact(
        name=model_name,
        type=type,
        description=model_description
    )
    artifact.add_file(f"./data/{model_name}.{file_extension}")
    wandb.log_artifact(artifact)

def load_artifact_path(artifact_name, version="latest", file_extension='pt'):
    artifact = wandb.use_artifact(f"{artifact_name}:{version}")
    directory = artifact.download()
    return f"{directory}/{artifact_name}.{file_extension}"