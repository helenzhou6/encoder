from torch import nn, no_grad, max, float32, Size
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from PIL import Image
from src.model.run_model import patch_image
from src.model.utils import get_device

def _predict_digit_using_model(tensor_digit, model, device):
    model.eval()
    with no_grad():
        input_tensor = tensor_digit.to(device)
        predicted_digit = model(input_tensor)
        result = predicted_digit.squeeze(0).argmax().item()
        return result

_transform_to_tensor = Compose([
    Resize((28, 28)),
    ToTensor(),
    Lambda(patch_image)
])

def _process_image(uint8_img):
    #  Process image from 280 pixel x 280 pixel, 4 colour channels (3 RGB + 1 alpha) uint8 to tensor (that is 28 x 28 and 1 greyscale channel)
    drawn_image = Image.fromarray(uint8_img)
    drawn_image_grey = drawn_image.convert("L") # convert to grayscale
    tensor_digit = _transform_to_tensor(drawn_image_grey)
    return tensor_digit.unsqueeze(0)

def predict_digit(uint8_img, model):
    tensor_digit = _process_image(uint8_img)
    device = get_device()
    return _predict_digit_using_model(tensor_digit, model, device)