# Encoder

Create a model that is trained on the MNIST dataset.
Takes in 28 pixel x 28 pixel image and predicts the digit (0 to 9). It also links it up with the frontend where it will 

![Example of it working](https://github.com/user-attachments/assets/d30e8ef0-3739-4d52-9b7a-c5849c75038b)

This is with a model that for evaluation had a loss of 0.08 & accuracy of 97%

## Architecture

This created the encoder architecture:

![encoder architecture](https://github.com/user-attachments/assets/a15de2f9-f256-4c48-b0e7-5b67b10b9c1c)

It's based on the 'Attention is all you need' 2017 paper, where it only uses the encoder part (the left hand side of their architecture diagram), and takes the output and finally puts it in the classifier.

This uses a single headed attention.
To see multi headed attention, see commit: https://github.com/helenzhou6/encoder/blob/85fe095765a8b26f02e6f0f7774fa3f1b64e3d65/src/model/init_model.py
And also here: https://github.com/besarthoxhaj/attention/blob/main/03_multihead_attention.py

To see the Encoder Decoder -- See the `espe` branch (see pull request). I might clone this into a separate repo later

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

### Model service/API
The model has been trained on the MNIST dataset. The code ensures that the model is usable by ensuring the model loss is < 0.5 and model accuracy is > 90% when testing on the MNIST testing dataset.
- To load the service locally, use `uv run uvicorn src.model.api:app`.
    - This runs the backend on port 8000, to check it is up and running go to: http://localhost:8000/healthcheck to see a response.

## Design choices
- There are different ways of writing the projections etc - see https://github.com/besarthoxhaj/attention/tree/main - but for readability opted for the simple one
- You can use sinusiodal (cosine) positional embeddings that adds a fixed param VS learned positional embeddings (see inside the class). The latter was chosen since good for fixed length vision tasks 