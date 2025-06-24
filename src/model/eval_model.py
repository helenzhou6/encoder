import os
from torch import inference_mode, nn, load
from torchvision import datasets
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

from utils import get_device
from init_model import SingleHeadAttentionModel
from run_model import NUM_CATEGORIES, patch_image

PATCH_SIZE = 7
EMBEDDING_DIM = PATCH_SIZE * PATCH_SIZE
NUM_PATCHES = 16
BATCH_SIZE = 32

device = get_device()
# TODO: FIX THE BELOW - download from wandb and run
model_pth_path = 'data/SingleHeadAttentionModel.pth'

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

if os.path.exists(model_pth_path):
    print(f"{model_pth_path} exists, loading model state...")
    model = SingleHeadAttentionModel(NUM_CATEGORIES, num_patches=NUM_PATCHES, dim_k=EMBEDDING_DIM).to(device)
    model.load_state_dict(load(model_pth_path), map_location=device)
else:
    print(f"{model_pth_path} does not exist - please run run_model.py to create...")

loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task = 'multiclass', num_classes=NUM_CATEGORIES).to(device)

loss, acc = 0, 0
model.eval()
with inference_mode():
    for X, y in test_dataloader:
        (y_pred, _) = model(X)
        loss += loss_fn(y_pred, y)
        acc += accuracy_fn(y_pred.argmax(dim=1), y)
    loss /= len(test_dataloader)
    acc /= len(test_dataloader)
    model_results_loss = loss.item()
    model_results_accuracy = (acc*100)
    
if model_results_loss > 0.5 or model_results_accuracy < 90:
    raise Exception(f"Machine learning model not usable - since model_loss was > 0.5 at {model_results_loss} and accuracy was < 90 at f{model_results_accuracy:.2f}%")
else:
    print(f"Model passed evaluation: model loss {model_results_loss} & {model_results_accuracy:.2f}%")