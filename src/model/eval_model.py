import os
from torch import inference_mode, nn, load
from torchvision import datasets
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

from utils import get_device, load_artifact_path, init_wandb
from init_model import LookerTransformer
from run_model import NUM_CATEGORIES, patch_image, NUM_PATCHES, INPUT_DIM, NUM_PATCHES, BATCH_SIZE, HIDDEN_DIM, DIMENSION_K, NUM_ENCODER_BLOCKS

device = get_device()
# TO RUN: make sure train_model() is commented out in run_model.py

init_wandb()
model_path = load_artifact_path("MultiHeadAttentionModel")

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(patch_image)
])

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform_image
)
test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if os.path.exists(model_path):
    print(f"{model_path} exists, loading model state...")
    model = LookerTransformer(output_shape=NUM_CATEGORIES, dim_input=INPUT_DIM, dim_hidden=HIDDEN_DIM, dim_k=DIMENSION_K, num_patches=NUM_PATCHES, num_encoder_blocks=NUM_ENCODER_BLOCKS).to(device)
    model.load_state_dict(load(model_path))
else:
    print(f"{model_path} does not exist - please run run_model.py to create...")

loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task = 'multiclass', num_classes=NUM_CATEGORIES).to(device)

loss, acc = 0, 0
model.eval()
with inference_mode():
    for X, y in test_dataloader:
        y_pred = model(X)
        loss += loss_fn(y_pred, y)
        acc += accuracy_fn(y_pred.argmax(dim=1), y)
    loss /= len(test_dataloader)
    acc /= len(test_dataloader)
    model_results_loss = loss.item()
    model_results_accuracy = (acc*100)
    
if model_results_loss > 0.5 or model_results_accuracy < 90:
    raise Exception(f"Machine learning model not usable - since model_loss was > 0.5 at {model_results_loss} and accuracy was < 90 at {model_results_accuracy:.2f}%")
else:
    print(f"Model passed evaluation: model loss {model_results_loss} & {model_results_accuracy:.2f}%")