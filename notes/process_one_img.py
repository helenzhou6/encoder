import torch
from torch import nn

EMBEDDING_DIM = 64
PATCH_SIZE = 7

linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)

def patch_image(image): # image = [1, 28, 28]
    patches = image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    print(patches.shape)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE) 
    print(patches.shape)
    return linear_proj(patches)

image = torch.randn(1, 28, 28)  
result = patch_image(image)
print(result.shape)