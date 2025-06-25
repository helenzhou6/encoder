
from torch import nn, Tensor, manual_seed
import torch
import math

class SingleHeadAttentionModel(nn.Module):
    def __init__(self, output_shape, dim_k=49):
        super().__init__()
        self.query_proj = nn.Linear(49, dim_k, bias=False)
        self.key_proj = nn.Linear(49, dim_k, bias=False)
        self.val_proj = nn.Linear(49, dim_k, bias=False)
        self.dim_k = dim_k
        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_k, output_shape)

    def forward(self, x):
        # Can't mat1 and mat2 shapes cannot be multiplied (512x64 and 49x49)
        Q = self.query_proj(x)  # Queries
        K = self.key_proj(x)  # Keys
        V = self.val_proj(x)  # Values
        
        # OVERALL: Calculates how much attention each patch should pay to every other patch
        dot_product_keys_queries = torch.matmul(Q, K.transpose(-2, -1)) # (batch_size, num_patch, num_patch) -- same code = qry @ key.transpose(-1, -2)
        sq_root_dim = (self.dim_k ** 0.5) # Mathematical stability so values not too large
        attention = dot_product_keys_queries / sq_root_dim
        normalised_attention = torch.nn.functional.softmax(attention, dim=-1) # -1 = last dimension (applied across the rows)
        hidden_v_representation = torch.matmul(normalised_attention, V)
        # TODO: add output x W = H as last step

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = hidden_v_representation.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output) 
        return logits, normalised_attention
