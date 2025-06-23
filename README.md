# Encoder

Create a model that is trained on the MNIST dataset
Takes in 28 pixel x 28 pixel image and predicts the digit (0 to 9)


## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site

## Code
1. `uv sync` download all dependencies needed to run project
2. Run `run_model.py` that will create the model, and train it. Also see wandb graphs. 
    - Run `wandb_login` in the terminal to enable wandb
    - This includes the Linear projection of flattened patches
3. `eval_model.py` will evaluate the model and check the loss and accuracy.