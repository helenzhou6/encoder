from fastapi import FastAPI, File, UploadFile
from numpy import frombuffer, uint8
import os
from torch import inference_mode, nn, load

from src.model.init_model import LookerTransformer
from src.model.run_model import NUM_CATEGORIES, patch_image, NUM_PATCHES, INPUT_DIM, NUM_PATCHES, BATCH_SIZE, HIDDEN_DIM, DIMENSION_K, NUM_ENCODER_BLOCKS
from src.model.utils import get_device, load_artifact_path, init_wandb
from src.model.inference import predict_digit

app = FastAPI()
device = get_device()
init_wandb()
model_path = load_artifact_path("MultiHeadAttentionModel")

if os.path.exists(model_path):
    print(f"{model_path} exists, loading model state...")
    model = LookerTransformer(output_shape=NUM_CATEGORIES, dim_input=INPUT_DIM, dim_hidden=HIDDEN_DIM, dim_k=DIMENSION_K, num_patches=NUM_PATCHES, num_encoder_blocks=NUM_ENCODER_BLOCKS).to(device)
    model.load_state_dict(load(model_path))
else:
    print(f"{model_path} does not exist - please run run_model.py to create...")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
      content = await file.read()
      uint8_img = frombuffer(content, dtype=uint8).reshape((280, 280, 4))
      predicted_digit = predict_digit(uint8_img, model)
      return {
            "predicted_digit": predicted_digit,
        }

@app.get("/healthcheck")
async def healthcheck():
      return "The model API is up and running"