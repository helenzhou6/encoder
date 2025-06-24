
from torch import nn
import torch

class SingleHeadAttentionModel(nn.Module):
    def __init__(self, output_shape, num_patches, dim_input, dim_k):
        super().__init__()
        self.dim_k = dim_k
        self.num_patches = num_patches
        # Combined projection for Q, K, V
        self.qkv_proj = nn.Linear(dim_input, dim_k * 3, bias=False)

        self.position_patch_embed = nn.Parameter(torch.zeros(1, num_patches, dim_k))
        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_k, output_shape)

    def forward(self, x):
        # Project to QKV and split into 3 tensors along last dimension
        # x is (batch_size, num_patches, dim_input)
        qkv = self.qkv_proj(x)  # shape: (batch_size, num_patches, dim_k * 3)
        Q, K, V = qkv.split(self.dim_k, dim=-1)  # each is (batch_size, num_patches, dim_k)

        Q = Q + self.position_patch_embed # (batch_size, num_patches, dim_k)
        K = K + self.position_patch_embed
        V = V + self.position_patch_embed

        attention_out = nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False) # shape: (batch_size, num_patches, dim_k)
        # TODO: add output x W = H as last step

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g.(mean pooling) take mean or sum over num_patches dimension
        pooled_output = attention_out.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output)  # (batch_size, output_shape)
        return logits, attention_out
