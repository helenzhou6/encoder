import os
from torch import inference_mode, nn, load
from torchvision import datasets
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import wandb

from utils import get_device
from init_model_mulhd import MultiHeadEncoderModel
from run_model import NUM_CATEGORIES, patch_image

device = get_device()
model_pth_path = 'model.pth'
BATCH_SIZE = 32

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

# ----- Load Model from W&B -----
wandb.init(
    project="DigitFinder",
    entity="week3-rebels",
    job_type="validation"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadEncoderModel(NUM_CATEGORIES, dim_k=48).to(device)

model_artifact = wandb.use_artifact(f"week3-rebels/DigitFinder/MultiHeadAttentionModel:v1", type="model")
artifact_dir = model_artifact.download()
model_path = os.path.join(artifact_dir, "week3-rebels/DigitFinder/MultiHeadAttentionModel:v1")
model.eval()


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