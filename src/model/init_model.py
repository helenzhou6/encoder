
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
# ALSO: Two linear layers with a non-linearity in between let your network learn non-linear, complex transformations, which a single linear layer cannot do by itself.
class FNN(torch.nn.Module):
    def __init__(self, dim_input):
        super().__init__()
        self.one = torch.nn.Linear(dim_input, dim_input)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.two = torch.nn.Linear(dim_input, dim_input)

    def forward(self, x):
        x = self.one(x)
        x = self.relu(x) # ReLU - introduce non-linearity
        x = self.dropout(x) # Random dropout 10%
        x = self.two(x)
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self, output_shape, dim_input, dim_k): # dim_input = main feature dimension, dim_k = dimension of Q & K
        super().__init__()
        self.attention = Attention(dim_input, dim_k)
        self.ffn = FNN(dim_input)
        self.initial_normalisation = torch.nn.LayerNorm(dim_input)
        self.final_normalisation = torch.nn.LayerNorm(dim_input)

        # TODO: REMOVE BELOW when we link it will the decoder
        self.classifier = nn.Linear(dim_input, output_shape)

    def forward(self, src):
        out = self.attention(src) # out = hidden_v
        # Residual connection = let model carry forward original input + transformed versio
        # -- Residual connections let each layer learn a residual function relative to its input, rather than a full new transformation.
        # make it easier to train and help gradient flow backward (adding more layers make training worse & in backpropagation gradients can disappear)
        src = src + out
        src = self.initial_normalisation(src)
        out = self.ffn(src)
        src = src + out
        src = self.final_normalisation(src)
        # return src

        # TODO: REMOVE THE BELOW when we link it will the decoder   
        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = src.mean(dim=1)  # (batch_size, dim_k)
        logits = self.classifier(pooled_output) # (batch_size, output_shape)
        return logits, src

