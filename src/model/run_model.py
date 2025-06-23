from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn, optim, inference_mode, save
from torchmetrics import classification, Accuracy
import wandb
from init_model import MNISTModel
from utils import init_wandb, get_device

device = get_device()
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.1

run = init_wandb()

class_names = datasets.MNIST(
    root="data",
    train=True,
    download=False,
).classes
classification_num=len(class_names)

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
train_dataloader = DataLoader(train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

input_shape = 784 
embedding_dim = 30

model = MNISTModel(input_shape, embedding_dim, classification_num, 
        ).to(device)

accuracy_fn = Accuracy(task = 'multiclass', num_classes=classification_num).to(device)
loss_fn = nn.CrossEntropyLoss()
model_optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

# -- TRAINING --

def train_model():
    for epoch in range(EPOCHS):
        print(f"---------\nTraining: Epoch {epoch + 1} out of {EPOCHS} ---------")
        train_loss, train_acc = 0, 0
        # y = classification
        for _, (batch_images, actual_y) in enumerate(train_dataloader):
            # Image is torch.Size([32, 1, 28, 28]) = batchsize, grayscale (1 channel) of 28px x 28px & is a torch.float32
            model.train()
            y_pred = model(batch_images)
            loss = loss_fn(y_pred, actual_y)
            train_loss += loss 
            train_acc += accuracy_fn(y_pred.argmax(dim=1), actual_y)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {(train_acc*100):.2f}%")
        run.log({"acc": train_acc*100, "loss": train_loss})

    save(model.state_dict(), 'model.pth')

# RUN below to train it
# train_model()
run.finish()