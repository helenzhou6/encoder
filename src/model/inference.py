from torch import nn, no_grad, max, float32, Size
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

def _predict_digit_using_model(tensor_digit, model):
    model.eval()
    with no_grad():
        predicted_digit = model(tensor_digit)
        print(predicted_digit)
        return predicted_digit

_transform_to_tensor = Compose([
    Resize((28, 28)),
    ToTensor(),
])

def _validate_digit_properties(tensor_digit):
    if tensor_digit.dtype is not float32:
          raise Exception(f"Incorrect type - tensor digit should be torch.float32, instead was {tensor_digit.dtype}")
    if tensor_digit.shape != Size([1, 28, 28]):
          raise Exception(f"Incorrect type - tensor digit should be torch.Size([1, 28, 28]), instead was {tensor_digit.shape}")
    
def _process_image(uint8_img):
    #  Process image from 280 pixel x 280 pixel, 4 colour channels (3 RGB + 1 alpha) uint8 to tensor (that is 28 x 28 and 1 greyscale channel)
    drawn_image = Image.fromarray(uint8_img)
    drawn_image_grey = drawn_image.convert("L") # convert to grayscale
    tensor_digit = _transform_to_tensor(drawn_image_grey)
    _validate_digit_properties(tensor_digit)
    return tensor_digit

def predict_digit(uint8_img, model):
    tensor_digit = _process_image(uint8_img)
    return _predict_digit_using_model(tensor_digit, model)