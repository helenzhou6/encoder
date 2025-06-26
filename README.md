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

## How to run decoder 
1. Run `multidigit_generator.py` and then `split_multidigit_dataset.py` to generate the datasets
2. Then run `train_encoder_decoder.py` to run the modelAdd commentMore actions

## Running sweeps
0. Make sure logged in - `export WANDB_API_KEY=<key>` if needed
1. Initialise: `wandb sweep src/model/sweep_config.yaml`
2. `wandb agent <copy and paste the yellow>`

## To Dos
1. Order of the prediction vs actual could be different order but still correct - adjust loss function to account for that
2. Add other hyperparams e.g. dim_k
3. For raw dataset being fed during training: add original MNIST/increase resolution of training dataset / fix cropped digits
3. Inference - Is it always confusing 1 and 9 etc - check. Could also increase the width of the stroke
4. [minor] Q: `pos_embed` - do we do it twice and if so can we remove it in one place?
