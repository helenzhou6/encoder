
from torch import nn
import torch

class AttentionModel(nn.Module):
    def __init__(self, output_shape, num_patches, dim_input, dim_k):
        super().__init__()
        self.dim_k = dim_k
        self.num_patches = num_patches
        self.position_patch_embed = nn.Parameter(torch.zeros(1, num_patches, dim_k))

        self.query_proj = nn.Linear(dim_input, dim_k, bias=False)
        self.key_proj = nn.Linear(dim_input, dim_k, bias=False)
        self.val_proj = nn.Linear(dim_input, dim_k, bias=False)
        self.out_proj = nn.Linear(dim_k, dim_input) # final projection
 
        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_k, output_shape)

    def forward(self, x):
        # Project to QKV and split into 3 tensors along last dimension
        # Split qkv into Q, K, V: each of shape (batch_size, seq_len, dim_k)
        # x: (batch_size, num_patches, dim_input)
        Q = self.query_proj(x)  # (batch_size, num_patches, dim_k)
        K = self.key_proj(x)    # (batch_size, num_patches, dim_k)
        V = self.val_proj(x)    # (batch_size, num_patches, dim_k)

        # Add positional embedding after projection
        Q = Q + self.position_patch_embed # (batch_size, num_patches, dim_k)
        K = K + self.position_patch_embed
        V = V + self.position_patch_embed

        dot_product_keys_queries = torch.matmul(Q, K.transpose(-2, -1))
        sq_root_dim = (self.dim_k ** 0.5)
        attention = dot_product_keys_queries / sq_root_dim

        normalised_attention = torch.softmax(attention, dim=-1)

        hidden_v_representation = torch.matmul(normalised_attention, V)
        # hidden_v_representation = self.out_proj(hidden_v_representation)
        # TODO: add output x W = H as last step

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = hidden_v_representation.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output) # (batch_size, output_shape)
        return logits, normalised_attention
