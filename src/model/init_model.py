
from torch import nn, manual_seed
import torch

class SingleHeadAttentionModel(nn.Module):
    def __init__(self, output_shape, num_patches, dim_k=49):
        super().__init__()
        self.query_proj = nn.Linear(49, dim_k, bias=False)
        self.key_proj = nn.Linear(49, dim_k, bias=False)
        self.val_proj = nn.Linear(49, dim_k, bias=False)
        self.dim_k = dim_k
        self.num_patches = num_patches
        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_k, output_shape)

    def forward(self, x):
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.dim_k))  # +1 for class token
        # x = patch_embeddings + self.pos_embed

        # Can't mat1 and mat2 shapes cannot be multiplied (512x64 and 49x49)
        Q = self.query_proj(x)  # Queries - shape (batch_size, num_patches, dim_k). Each row is query vector for a patch
        K = self.key_proj(x)  # Keys - shape (batch_size, num_patches, dim_k). Each row is a key vector for a patch
        V = self.val_proj(x)  # Values
        
        # OVERALL: Calculates how much attention each patch should pay to every other patch
        # -- by comparing the query (Q) of each patch to the keys (K) of all other patches.
        # 1. Applies dot product of queries and keys (same code as qry @ key.transpose(-1, -2)). Transpose flips axis to (batch_size, dim_k, num_patches) so dot product/matrix multiplication can be applied
        dot_product_keys_queries = torch.matmul(Q, K.transpose(-2, -1)) # (batch_size, num_patch, num_patch)
        # 2. Divide by square root of dimensin of K for Mathematical stability so attention is not skewed because of size of embeddings
        sq_root_dim = (self.dim_k ** 0.5)
        attention = dot_product_keys_queries / sq_root_dim # (batch_size, num_patch, num_patch)
        # 3b(optional). Add mask to prevent "looking ahead" - mask fills the upper triangle of matrix multiplication with -inf, so softmax turns those into zero.
        # - torch.ones creates a matrix of 1s, size (num_patches x num_patches), tril sets upper triangle to 0 (rest are 1), ==0 converts to Boolean (True = future positions to mask), .masked_fill - where mask is True, replces the value with -inf
        mask = torch.tril(torch.ones(self.num_patches, self.num_patches)) == 0
        attention = attention.masked_fill(mask, -float('inf'))
        # 3a. Apply softmax to turn scores to probabilities 
        normalised_attention = torch.softmax(attention, dim=-1) # -1 = last dimension (applied across the rows)
        # 4. Apply attention weights to the values (i.e. vector) to get final output
        hidden_v_representation = torch.matmul(normalised_attention, V)
        # TODO: add output x W = H as last step

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = hidden_v_representation.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output) 
        return logits, normalised_attention
