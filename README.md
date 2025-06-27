# Decoder Encoder i.e. Transformer
Creates a Transformer model that is trained on the MNIST dataset - that has been added all together into either 1, 2, 3 or 4 digits all on the same digit. The encoder-decoder will try and predict all the digits

## Architecture
This uses the transformer architecture (as defined by the Attention is All you need 2017 paper). This is taken from there:
![Architecture](https://github.com/user-attachments/assets/735a455a-810f-4ced-a1ca-d3200f4aa65b)

![Other architecture](https://github.com/user-attachments/assets/5f8c69b8-9d48-4839-a3a6-dd830ebb3856)

i.e. it consists of:
1. Cut the image into patches and add positional encoding
2. **Encoder**: multiple encoder layer/blocks that each has a multi headed attention + FNN (a linear layer, ReLU and linear layer)
3. **Decoder** that has: multiple decoder layer/blocks that each has a multi headed self attention (with masking of the future) + multi headed cross attention (that takes the input from the Encoder) + FNN (a linear layer, ReLU and linear layer)
4. **Classification**: The final layer that does the generation of the sequence of digits it thinks has been drawn at inference (during training, it will compare with the actual sequence to learn)

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site

## How to run encoder - decoder 
0. `uv sync` download all dependencies needed to run project
1. Run `multidigit_generator.py` and then `split_multidigit_dataset.py` to generate the datasets (you might need to set `download=True` if you don't already have the MNIST dataset downloaded locally)
2. Then run `train_encoder_decoder.py` to run the model, make sure `run_config["run_type"] = "train"`
    - Run `wandb_login` in the terminal to enable wandb, or `export WANDB_API_KEY=<key>` 

## Running sweeps
0. Make sure logged into wandb - `export WANDB_API_KEY=<key>` if needed
1. Run `train_encoder_decoder` & ensure `run_config["run_type"] = "sweep"`. You can change the config to `sweep_config.py`

## Running the frontend
- Run `uv run streamlit run front_end.py`

## To Dos
1. Order of the prediction vs actual could be different order but still correct - adjust loss function to account for that
2. Add other hyperparams e.g. dim_k
3. For raw dataset being fed during training: add original MNIST/increase resolution of training dataset / fix cropped digits
3. Inference - Is it always confusing 1 and 9 etc - check. Could also increase the width of the stroke?
