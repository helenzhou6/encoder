
from torch import nn
import torch
import matplotlib.pyplot as plt

def visualise_attention(normalised_attention):
    attention_matrix = normalised_attention[0]
    if attention_matrix.ndim == 3:
        attention_matrix = attention_matrix[0]
    attention_matrix = attention_matrix.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.matshow(attention_matrix, cmap='viridis', fignum=1)
    plt.title("Attention map")
    plt.xlabel("One Key patch")
    plt.ylabel("One Query patch") 
    plt.colorbar()
    plt.show()

class SingleHeadAttentionModel(nn.Module):
    def __init__(self, output_shape, num_patches, dim_k=49):
        super().__init__()
        self.query_proj = nn.Linear(49, dim_k, bias=False)
        self.key_proj = nn.Linear(49, dim_k, bias=False)
        self.val_proj = nn.Linear(49, dim_k, bias=False)
        self.dim_k = dim_k
        self.num_patches = num_patches

        self.position_patch_embed = nn.Parameter(torch.zeros(1, num_patches, dim_k))
        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_k, output_shape)

    def forward(self, x):
        x = x + self.position_patch_embed
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.val_proj(x)
        
        dot_product_keys_queries = torch.matmul(Q, K.transpose(-2, -1))
        sq_root_dim = (self.dim_k ** 0.5)
        attention = dot_product_keys_queries / sq_root_dim

        normalised_attention = torch.softmax(attention, dim=-1)
        # visualise_attention(normalised_attention)

        hidden_v_representation = torch.matmul(normalised_attention, V)
        # TODO: add output x W = H as last step

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = hidden_v_representation.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output) 
        return logits, normalised_attention
