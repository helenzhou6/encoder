import os
from torch import inference_mode, nn, load
from torchvision import datasets
from torchmetrics import classification, Accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from utils import get_device
from init_model import MNISTModel
from run_model import input_shape, hidden_units, classification_num

device = get_device()
model_pth_path = 'model.pth'
BATCH_SIZE = 32

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -- Loads the model
if os.path.exists(model_pth_path):
    print("model.pth exists, loading model state...")
    model = MNISTModel(input_shape,
        hidden_units,
        classification_num
    ).to(device)
    model.load_state_dict(load('model.pth'))
else:
    print("model.pth does not exist - please run run_model.py to create...")

loss_fn = nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task = 'multiclass', num_classes=classification_num).to(device)

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
    raise Exception(f"Machine learning model not usable - since model_loss was > 0.5 at {model_results_loss} and accuracy was < 90 at {model_results_accuracy}")
else:
    print(f"Model passed evaluation: model loss {model_results_loss} & {model_results_accuracy:.2f}%")