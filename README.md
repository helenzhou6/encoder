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

# To Dos
1. [In progress] Change loss_fn to also ignore 11 token. Might also impact the accuracy (since later step)
Tried but didn't work:
```python
pred_y = logits.view(-1, VOCAB_SIZE)
actual_y = tgt_output.view(-1)
loss = loss_fn(pred_y, actual_y) # shape: (N,) if reduction='none'
mask = (actual_y != 11) & (actual_y != 12) # ignore <pad> and <eos>
loss = (loss * mask).sum() / mask.sum()
```
2. Order of the prediction vs actual could be different order but still correct - adjust loss function to account for that
3. Is it always confusing 1 and 9 etc - check
4. [minor] Q: `pos_embed` - do we do it twice and if so can we remove it in one place?