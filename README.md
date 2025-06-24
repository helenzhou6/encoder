# Encoder

Create a model that is trained on the MNIST dataset
Takes in 28 pixel x 28 pixel image and predicts the digit (0 to 9)


## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site
- You may need to run `export PYTHONPATH=./src` if it doesn't recognise imports from within the codebase

## Code
1. `uv sync` download all dependencies needed to run project
    - To set the venv in VSCode (on Mac, you can do Shift Command P - you can select the interpreter to be virtual env uv has set up, in .venv). Or run `uv run <script>` to run.
2. Run `run_model.py` that will create the model, and train it. Also see wandb graphs. 
    - Run `wandb_login` in the terminal to enable wandb
    - This includes the Linear projection of flattened patches
3. `eval_model.py` will evaluate the model and check the loss and accuracy.

## To Do
1. ✅ Add the positional embedding
2. ✅ Make multiple encoder blocks
3. Add decoder
4. Train with multiple numbers
5. Sweeps - check best hyperparams & also upload attention maps wandb
6. Frontend / inference

## Design choices
- There are different ways of writing the projections etc - see https://github.com/besarthoxhaj/attention/tree/main - but for readability opted for the simple one
- You can use sinusiodal (cosine) positional embeddings that adds a fixed param VS learned positional embeddings (see inside the class). The latter was chosen since good for fixed length vision tasks 