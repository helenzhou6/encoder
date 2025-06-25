from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch import nn, optim, save

# from plot_attention import visualise_attention
from utils import get_device, init_wandb, save_artifact
from init_model import LookerTransformer

PATCH_SIZE = 7
INPUT_DIM = PATCH_SIZE * PATCH_SIZE
NUM_PATCHES = 16
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.1
# Query and key vectors are projected into a lower-dimensional space (i.e., dim_k < input_dim) for efficiency and generalization.
DIMENSION_K = 32
NUM_ENCODER_BLOCKS = 6

device = get_device()
wandb_run = init_wandb()

# -- Chop image into patches
linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, INPUT_DIM)
def patch_image(image): # image = [1, 28, 28]
    assert image.shape == (1, 28, 28), f"Expected (1,28,28) but got {image.shape}"
    patches = image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE) 
    return linear_proj(patches)
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
model = LookerTransformer(output_shape=NUM_CATEGORIES, dim_input=INPUT_DIM, dim_hidden=HIDDEN_DIM, dim_k=DIMENSION_K, num_patches=NUM_PATCHES, num_encoder_blocks=NUM_ENCODER_BLOCKS).to(device)

accuracy_fn = Accuracy(task = 'multiclass', num_classes=NUM_CATEGORIES).to(device)
loss_fn = nn.CrossEntropyLoss()
model_optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

def train_model():
    for epoch in range(EPOCHS):
        print(f"----\nTraining: Epoch {epoch + 1} out of {EPOCHS} ----")
        train_loss, train_acc = 0, 0
        # y = classification
        for _, (batch_images, actual_y) in enumerate(train_dataloader):
            # batch_images is [32, 16, 49] - [B, num_patch, input_dim]
            model.train()
            y_pred = model(batch_images)
            loss = loss_fn(y_pred, actual_y)
            train_loss += loss 
            train_acc += accuracy_fn(y_pred.argmax(dim=1), actual_y)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            # if batch_idx == 0:
                # Only first patch to reduce noise
                # visualise_attention(attention_weights, step=epoch)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {(train_acc*100):.2f}%")
        wandb_run.log({"acc": train_acc*100, "loss": train_loss})

    model_path = "data/MultiHeadAttentionModel.pt"
    model_state = model.state_dict()
    save(model_state, model_path)
    save_artifact(
        "MultiHeadAttentionModel",
        "Multi headed attention model"
    )

# Uncomment below to train it
# train_model()
wandb_run.finish()