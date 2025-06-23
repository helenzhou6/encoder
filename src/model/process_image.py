from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

EMBEDDING_DIM = 64
PATCH_SIZE = 7
BATCH_SIZE = 32

linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)

def patch_image(image): # image = [1, 28, 28]
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

for _, (batch_images, _) in enumerate(train_dataloader):
    print(batch_images.shape)
