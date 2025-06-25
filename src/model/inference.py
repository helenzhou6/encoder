from torch import no_grad, load
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, Normalize, Grayscale
from PIL import Image
import os
from run_model import patch_image
from utils import get_device, init_wandb, load_artifact_path
from init_model import LookerTransformer
from run_model import NUM_CATEGORIES, INPUT_DIM, HIDDEN_DIM, DIMENSION_K, NUM_PATCHES, NUM_ENCODER_BLOCKS

device = get_device()

init_wandb()
model_path = load_artifact_path("MultiHeadAttentionModel")
if os.path.exists(model_path):
    print(f"{model_path} exists, loading model state...")
    model = LookerTransformer(output_shape=NUM_CATEGORIES, dim_input=INPUT_DIM, dim_hidden=HIDDEN_DIM, dim_k=DIMENSION_K, num_patches=NUM_PATCHES, num_encoder_blocks=NUM_ENCODER_BLOCKS).to(device)
    model.load_state_dict(load(model_path))
    print("Loaded to model")
else:
    print(f"{model_path} does not exist - please run run_model.py to create...")

def _predict_digit_using_model(tensor_digit, model, device):
    model.eval()
    with no_grad():
        input_tensor = tensor_digit.to(device)
        logits = model(input_tensor)
        return logits.argmax(dim=1).item()

_transform_to_tensor = Compose([
    Resize((28, 28)),
    ToTensor(),
    Lambda(patch_image),
])

def _process_image(image):
    #  Process image from 280 pixel x 280 pixel, 4 colour channels (3 RGB + 1 alpha) uint8 to tensor (that is 28 x 28 and 1 greyscale channel)
    # drawn_image = Image.fromarray(uint8_img)
    tensor_digit = _transform_to_tensor(image)
    return tensor_digit.unsqueeze(0)

def predict_digit(image):
    tensor_digit = _process_image(image)
    
    return _predict_digit_using_model(tensor_digit, model, device)