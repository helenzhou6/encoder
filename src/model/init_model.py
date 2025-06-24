
from torch import nn
import torch

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, output_shape, num_patches, dim_input, dim_k, num_heads):
        super().__init__()
        self.dim_k = dim_k
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.head_dim = dim_k // num_heads
        assert dim_k % num_heads == 0, "dim_k must be divisible by num_heads"
        # Combined projection for Q, K, V
        self.qkv_proj = nn.Linear(dim_input, dim_k * 3, bias=False)

        self.position_patch_embed = nn.Parameter(torch.zeros(1, num_patches, dim_k))

        # Optional final projection after concatenating heads
        self.out_proj = nn.Linear(dim_k, dim_k)

        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_k, output_shape)

    def forward(self, x):
        # Project to QKV and split into 3 tensors along last dimension
        # x is (batch_size, num_patches, dim_input)
        qkv = self.qkv_proj(x)  # shape: (batch_size, num_patches, dim_k * 3)
        # Split qkv into Q, K, V: each of shape (batch_size, seq_len, dim_k)
        Q, K, V = qkv.chunk(3, dim=-1)
        batch_size = x.size(0)

        Q = Q + self.position_patch_embed # (batch_size, num_patches, dim_k)
        K = K + self.position_patch_embed
        V = V + self.position_patch_embed

        def reshape_for_heads(tensor):
            return tensor.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        Q = reshape_for_heads(Q)
        K = reshape_for_heads(K)
        V = reshape_for_heads(V)

        # Scaled dot-product attention
        attn_scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, heads, N, N)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ V  # (B, heads, N, head_dim)

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, self.num_patches, self.dim_k)  # (B, N, dim_k)

        # Optional output projection
        attn_out = self.out_proj(attn_out)  # (B, N, dim_k)

        # Pool and classify
        pooled_output = attn_out.mean(dim=1)  # (B, dim_k)
        logits = self.classifier(pooled_output)  # (B, output_shape)

        return logits, attn_weights
