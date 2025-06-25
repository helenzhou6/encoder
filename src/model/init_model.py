
from torch import nn
import torch

# Self-attention layer - captures relationships (i.e. how tokens - image patches - relate to each other)
# Attention = weighted average of values -- linear in nature (lacks non-linear) and not apply complex transformations 
class Attention(nn.Module):
    def __init__(self, dim_input, dim_k):
        super().__init__()
        self.dim_k = dim_k
        self.query_proj = nn.Linear(dim_input, dim_k, bias=False)
        self.key_proj = nn.Linear(dim_input, dim_k, bias=False)
        self.val_proj = nn.Linear(dim_input, dim_k, bias=False)
        self.drpout = torch.nn.Dropout(0.1)
        self.out_proj = nn.Linear(dim_k, dim_input) # final projection

    def forward(self, x):
        # Project to QKV and split into 3 tensors along last dimension
        # Split qkv into Q, K, V: each of shape (batch_size, seq_len, dim_k)
        # x: (batch_size, num_patches, dim_input)
        Q = self.query_proj(x)  # (batch_size, num_patches, dim_k)
        K = self.key_proj(x)    # (batch_size, num_patches, dim_k)
        V = self.val_proj(x)    # (batch_size, num_patches, dim_k)
        attention = Q @ K.transpose(-2, -1) * self.dim_k ** -0.5
        attention = torch.softmax(attention, dim=-1)
        attention = self.drpout(attention)
        out = torch.matmul(attention, V)
        return self.out_proj(out)
    
# EncoderLayer - need FNN to apply nonlinear transformation to each patch embedding - so model can do more complex functions
# -- works independently on each token
# -- without FNN- would be doing a soft mixing of features, but not actually learning to transform or process those features individually.
class EncoderLayer(torch.nn.Module):
    def __init__(self, output_shape, dim_input, dim_k):
        super().__init__()
        self.attention = Attention(dim_input, dim_k)
        self.initial_normalisation = torch.nn.LayerNorm(dim_input)
        self.final_normalisation = torch.nn.LayerNorm(dim_input) # 

        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_input, output_shape)

    def forward(self, src):
        out = self.attention(src) # out = hidden_v
        src = src + out
        src = self.initial_normalisation(src)
        # out = self.ffn(src)
        # src = src + out
        src = self.final_normalisation(src)
        # return src

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = src.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output) # (batch_size, output_shape)
        return logits, src

