import torch
from torch import nn, optim, save
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import wandb
import argparse

from utils import get_device, init_wandb, save_artifact
from init_model_mulhd import MultiHeadEncoderModel

# ONLY runs the MultiHeadEncoderModel (not the decoder)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.1

PATCH_SIZE = 7
EMBEDDING_DIM = 48
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.1

device = get_device()
wandb_run = init_wandb()

parser = argparse.ArgumentParser()
parser.add_argument("--num_blocks", type=int, default=2, help="Number of encoder blocks")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
args = parser.parse_args()

NUM_ENCODER_BLOCKS = args.num_blocks
NUM_HEADS = args.num_heads

class PositionalEncoding2D(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        #One half (dim // 2) is used for the row embedding, The other half (dim // 2) is used for the column embedding.
        self.row_embed = nn.Embedding(height, dim // 2)
        self.col_embed = nn.Embedding(width, dim // 2)
        self.height = height
        self.width = width

    def forward(self):
        rows = torch.arange(self.height)
        cols = torch.arange(self.width)
        pos = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)  # [H, W, 2]
        pos = pos.reshape(-1, 2)  # [H*W, 2]
        row_pos = self.row_embed(pos[:, 0])
        col_pos = self.col_embed(pos[:, 1])
        return torch.cat([row_pos, col_pos], dim=1)  # [H*W, dim]

linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)

def patch_image(image):
    patches = image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE)
    return linear_proj(patches)

transform_image = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Lambda(patch_image)
])

train_data = datasets.MNIST(
    root=".data",
    train=True,
    download=False,
    transform=transform_image
)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
NUM_CATEGORIES = len(train_data.classes)

model = MultiHeadEncoderModel(
    num_classes=NUM_CATEGORIES,
    dim_k=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_blocks=NUM_ENCODER_BLOCKS,
    pos_embed=PositionalEncoding2D(4, 4, EMBEDDING_DIM).to(device)
).to(device)

accuracy_fn = Accuracy(task='multiclass', num_classes=NUM_CATEGORIES).to(device)
loss_fn = nn.CrossEntropyLoss()
model_optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

def train_model():
    for epoch in range(EPOCHS):
        print(f"----\nTraining: Epoch {epoch + 1} out of {EPOCHS} ----")
        train_loss, train_acc = 0, 0
        for _, (batch_images, actual_y) in enumerate(train_dataloader):
            model.train()
            y_pred = model(batch_images.to(device))
            loss = loss_fn(y_pred, actual_y.to(device))
            train_loss += loss.item()
            predictions = y_pred.detach()
            train_acc += accuracy_fn(predictions.argmax(dim=1), actual_y.to(device)).item()
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {(train_acc*100):.2f}%")
        wandb_run.log({"acc": train_acc*100, "loss": train_loss})

    model_path = ".data/MultiHeadAttentionModel-1block-4head.pt"
    save(model.state_dict(), model_path)
    save_artifact("MultiHeadAttentionModel", "MultiHeadAttentionModel")

train_model()
wandb_run.finish()
