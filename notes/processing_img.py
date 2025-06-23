import torch
import torch.nn as nn

# Batch of grayscale images: (batch_size, channels, height, width)
images = torch.randn(32, 1, 28, 28)  

# DIMENSIONS: = (batch_size, channels, height, width)
# shape: [32, 1, 28, 28] = batch size of 32, 1 color channel (grayscale), 28px height by 28px width

PATCH_SIZE = 7

# Step 1: Split into non-overlapping 7x7 patches using unfold
# Unfold on 2nd dimension (HEIGHT) will create sliding windows of size 7 over the height, and result in:
pre_patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE) 
# print(pre_patches.shape) 

# SO output is torch.Size([32, 1, 4, 28, 7]) =  Shape: [batch, channels, num_patches_H, full_width, patch_height]
# You have 4 vertical strips (patches) per image.
# Each strip is 7 pixels high and spans the full width (28).

# THEN
# Unfold on 3rd dimension (WIDTH) will create sliding windows over the height, and result in:
patches = pre_patches.unfold(3, PATCH_SIZE, PATCH_SIZE)
# print(patches.shape) torch.Size([32, 1, 4, 4, 7, 7]) = shape: (batch_size, channels, num_patches_H, num_patches_W, patch_size, patch_size)

# # Step 2: Rearrange and flatten
# Change shape to (batch_size, num_patches, patch_size * patch_size)
# contiguous() = makes the tensor memory-contiguous to ensure memory layout is clean
clean_memory_patches = patches.contiguous()
flatten_patches = clean_memory_patches.view(32, 1, -1, PATCH_SIZE * PATCH_SIZE)  # (32, 1, 16, 49) = (batch size, num_channels, num_patches 16 total patches per image, flatten_pixels = flattening each patch from shape 7,7 to 1D vector of length 49)
# Reshape this tensor into (32, 1, some_number, 49), and figure out some_number automatically (with -1), so that the total number of elements stays the same.
# -1 = So some_number must be 784 / 49 = 16 

# Don't need the channel dim, so remove
patches = flatten_patches.squeeze(1)  # (32, 16, 49)
# Result for the patch = is the raw pixel data, not a learned representation

# # Step 3: Linear projection of each patch - which makes it a 64D embedding vector and this creates a learnable linear layer
EMBEDDING_DIM = 64  # Embedding dimension for transformer input
linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM) # This makes it a 64 dimension embedding

# # Apply linear layer to each patch
embedded_patches = linear_proj(patches)  # shape: (32, 16, 64) = (32 batches, 16 num_patches, 64 embedding dimension)
print(embedded_patches.shape)

# In a transformer model, inputs are expected to be a sequence of vectors, where:
# - Each vector is a token embedding (just like a word embedding in NLP)
# - Each token should have the same embedding size (e.g., 64, 128, 768...)

# Just like:
# token_embedding = nn.Embedding(vocab_size, embedding_dim)
# ...which maps a discrete word ID into a dense vector, this linear projection maps a flat patch of raw pixel values into a dense, learnable vector that captures richer information.