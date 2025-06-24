import torch
import math

# Set random seed for reproducibility
torch.manual_seed(42)

# Parameters
batch_size = 1
num_patches = 5           # number of patches or tokens
embed_dim = 8             # input embedding dimension
num_heads = 2             # number of attention heads
dim_k = 12                # total dimension for Q, K, V (must be divisible by num_heads)
head_dim = dim_k // num_heads  # dimension per head

assert dim_k % num_heads == 0, "dim_k must be divisible by num_heads"

# Input tensor (batch_size, seq_len, embed_dim)
x = torch.rand(batch_size, num_patches, embed_dim)

# Linear layer to project to Q, K, V (output: 3 * dim_k)
qkv_proj = torch.nn.Linear(embed_dim, dim_k * 3)
qkv = qkv_proj(x)  # shape: (batch_size, seq_len, dim_k * 3)

# Split qkv into Q, K, V: each of shape (batch_size, seq_len, dim_k)
qry, key, val = qkv.chunk(3, dim=-1)

# Reshape to (batch_size, num_heads, seq_len, head_dim)
def reshape_for_heads(tensor):
    return tensor.reshape(batch_size, num_patches, num_heads, head_dim).transpose(1, 2)

qry = reshape_for_heads(qry)
key = reshape_for_heads(key)
val = reshape_for_heads(val)

# Compute scaled dot-product attention
att = qry @ key.transpose(-1, -2) / math.sqrt(head_dim)  # (batch_size, num_heads, seq_len, seq_len)
out = (att @ val).transpose(1, 2).reshape(batch_size, num_patches, dim_k)  # (batch_size, seq_len, dim_k)

# Optionally: project output back to embed_dim if needed
# out_proj = torch.nn.Linear(dim_k, embed_dim)
# out = out_proj(out)

# Print shapes for verification
print(f"Input x shape: {x.shape}")
print(f"Q shape after reshape: {qry.shape}")
print(f"Attention weights shape: {att.shape}")
print(f"Output shape: {out.shape}")