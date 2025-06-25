from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch import nn, optim, save
import wandb

from utils import get_device, init_wandb, save_artifact
from init_model import SingleHeadAttentionModel

PATCH_SIZE = 7
EMBEDDING_DIM = PATCH_SIZE * PATCH_SIZE
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.1
device = get_device()

wandb_run = init_wandb()

# -- Chop image into patches

linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)
# row_embed: stores 4 vectors for the 4 patch rows, col_embed: stores 4 vectors for the 4 patch columns
row_embed = nn.Embedding(4, EMBEDDING_DIM // 2)
col_embed = nn.Embedding(4, EMBEDDING_DIM // 2)

def patch_image(image):  # image = [1, 28, 28]
    #splits the image into non-overlapping patches of size 7×7
    patches = image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE) # [1, 4, 4, 7, 7]
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE)
    patch_embeddings = linear_proj(patches) #Each 49-element patch vector is passed through the linear_proj layer to produce an embedding
    # patch_embeddings = [32, 16, 49] where 32 is batch size, 16 is number of patches, and 49 is embedding dimension
    # 2D positional encoding for 4x4 = 16 patches
    positions = torch.arange(16, device=patch_embeddings.device)
    #Computes the row index and column index for each patch
    rows = positions // 4
    cols = positions % 4
    pos_embed = torch.cat([row_embed(rows), col_embed(cols)], dim=1)

    patch_embeddings = patch_embeddings + pos_embed
    return patch_embeddings

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(patch_image)
])

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_image,
    target_transform=None
)

train_dataloader = DataLoader(train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

NUM_CATEGORIES = len(train_data.classes)

model = SingleHeadAttentionModel(NUM_CATEGORIES, dim_k=EMBEDDING_DIM).to(device)

accuracy_fn = Accuracy(task = 'multiclass', num_classes=NUM_CATEGORIES).to(device)
loss_fn = nn.CrossEntropyLoss()
model_optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

def train_model():
    for epoch in range(EPOCHS):
        print(f"----\nTraining: Epoch {epoch + 1} out of {EPOCHS} ----")
        train_loss, train_acc = 0, 0
        # y = classification
        for _, (batch_images, actual_y) in enumerate(train_dataloader):
            # batch_images is [32, 16, 49]git 
            model.train()
            (y_pred, _) = model(batch_images)
            loss = loss_fn(y_pred, actual_y)
            train_loss += loss 
            train_acc += accuracy_fn(y_pred.argmax(dim=1), actual_y)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {(train_acc*100):.2f}%")
        wandb_run.log({"acc": train_acc*100, "loss": train_loss})

    model_path = "data/SingleHeadAttentionModel.pt"
    model_state = model.state_dict()
    save(model_state, model_path)
    save_artifact(
        "SingleHeadAttentionModel",
        "SingleHeadAttentionModel"
    )

# Uncomment below to train it
train_model()
wandb_run.finish()