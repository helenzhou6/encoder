
from torch import nn, Tensor, manual_seed
import torch

class SingleHeadAttentionModel(nn.Module):
    def __init__(self, output_shape, dim_k=49):
        super().__init__()
        self.W_Q = nn.Linear(49, dim_k, bias=False)
        self.W_K = nn.Linear(49, dim_k, bias=False)
        self.W_V = nn.Linear(49, dim_k, bias=False)
        self.dim_k = dim_k
        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_k, output_shape)

    def forward(self, x):
        # Can't mat1 and mat2 shapes cannot be multiplied (512x64 and 49x49)
        Q = self.W_Q(x)  # Queries
        K = self.W_K(x)  # Keys
        V = self.W_V(x)  # Values
        
        # Calculate how much attention each patch should pay to every other patch
        dot_product_keys_queries = torch.matmul(Q, K.transpose(-2, -1)) # (batch_size, num_patch, num_patch)
        sq_root_dim = (self.dim_k ** 0.5) # Mathematical stability so values not too large
        attention_scores = dot_product_keys_queries / sq_root_dim
        weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(weights, V)
        # TODO: add output x W = H as last step

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = attention_output.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output) 
        return logits, weights
