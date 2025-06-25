# Encoder

Create a model that is trained on the MNIST dataset
Takes in 28 pixel x 28 pixel image and predicts the digit (0 to 9)


## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site
- You may need to run `export PYTHONPATH=./src` if it doesn't recognise imports from within the codebase

Run `uv sync` download all dependencies needed to run project
    - To set the venv in VSCode (on Mac, you can do Shift Command P - you can select the interpreter to be virtual env uv has set up, in .venv). Or run `uv run <script>` to run.

## Running the model
2. Run `run_model.py` that will create the model, and train it. Also see wandb graphs. 
    - Run `wandb_login` in the terminal to enable wandb
    - This includes the Linear projection of flattened patches
3. `eval_model.py` will evaluate the model and check the loss and accuracy.

## Streamlit Front end
Then run the script: `streamlit run src/frontend/app.py` and it will create a localhost URL to view. 

## To Do
1. ✅ Add the positional embedding
2. ✅ Change code to be like Bes' - https://github.com/mlx-fac/vit
3. Make multiple encoder blocks
4. Sweeps - check best hyperparams

### Decoder bits
4. Add decoder
5. Train with multiple numbers
6. Frontend / inference

## Design choices
- There are different ways of writing the projections etc - see https://github.com/besarthoxhaj/attention/tree/main - but for readability opted for the simple one
- You can use sinusiodal (cosine) positional embeddings that adds a fixed param VS learned positional embeddings (see inside the class). The latter was chosen since good for fixed length vision tasks 