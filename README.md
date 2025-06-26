# Encoder
Create a model that is trained on the MNIST dataset - that has been added all together into either 1, 2, 3 or 4 digits all on the same digit. The encoder-decoder will try and predict all the digits

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site

## How to run encoder - decoder 
0. `uv sync` download all dependencies needed to run project
1. Run `multidigit_generator.py` and then `split_multidigit_dataset.py` to generate the datasets (you might need to set `download=True` if you don't already have the MNIST dataset downloaded locally)
2. Then run `train_encoder_decoder.py` to run the model
    - Run `wandb_login` in the terminal to enable wandb, or `export WANDB_API_KEY=<key>` 

## Running sweeps
0. Make sure logged in - `export WANDB_API_KEY=<key>` if needed
1. Initialise: `wandb sweep src/model/sweep_config.yaml`
2. `wandb agent <copy and paste the yellow>`

## Running the frontend
- Run `uv run streamlit run front_end.py`

## To Dos
1. Order of the prediction vs actual could be different order but still correct - adjust loss function to account for that
2. Add other hyperparams e.g. dim_k
3. For raw dataset being fed during training: add original MNIST/increase resolution of training dataset / fix cropped digits
3. Inference - Is it always confusing 1 and 9 etc - check. Could also increase the width of the stroke
4. [minor] Q: `pos_embed` - do we do it twice and if so can we remove it in one place?
