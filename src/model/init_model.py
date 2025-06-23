
from torch import nn, Tensor, manual_seed, load
from torchvision import datasets

manual_seed(42)
class MNISTModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            # TODO: Change it so that the input will not be 784 pixels, but patches of the images
            nn.Flatten(), # Converts 2D image data [batch, 1 (channel), 28 (px height), 28 (px width)] into 1D vector of shape [batch, 784]
            nn.Linear(in_features=input_shape, out_features=hidden_units), # Return vectors of embedding size 30
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape), # 30 -> to classification 0 to 9
            nn.ReLU()
        )
    def forward(self, x: Tensor):
        return self.layer_stack(x)
