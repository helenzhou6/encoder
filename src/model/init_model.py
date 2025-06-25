
from torch import nn
import torch

# Self-attention layer - captures relationships (i.e. how tokens - image patches - relate to each other)
# Attention = weighted average of values -- linear in nature (lacks non-linear) and not apply complex transformations 
# -- Note, this is single headed. Have multiple attention heads would look at the input from different perspectives (e.g. edges and shapes)
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

# FNN need to apply nonlinear transformation to each patch embedding - so model can do more complex functions
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
    def __init__(self, dim_input, dim_k): # dim_input = main feature dimension, dim_k = dimension of Q & K
        super().__init__()
        self.attention = Attention(dim_input, dim_k)
        self.ffn = FNN(dim_input)
        self.initial_normalisation = torch.nn.LayerNorm(dim_input)
        self.final_normalisation = torch.nn.LayerNorm(dim_input)

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
        return src

# Having multiple encoder layers (blocks) - applied attention + FNN logic to output of previous layer, stacking adds depth. High layers = more complex/gloval dependencies vs lower layers = simple/local patterns
class LookerTransformer(torch.nn.Module):
    def __init__(self, output_shape, dim_input, dim_hidden, dim_k, num_patches, num_encoder_blocks):  # <- input = 49, hidden = 128
        super().__init__()
        # self.cls = torch.nn.Parameter(torch.randn(1, 1, dim_hidden))           # [1, 1, dim_hidden]
        self.embedding = torch.nn.Linear(dim_input, dim_hidden)                  # [dim_input → dim_hidden]
        self.num_tokens = num_patches                                           # = num of patches + 1 class token
        self.position_emb = torch.nn.Embedding(self.num_tokens, dim_hidden)    
        self.register_buffer('rng', torch.arange(self.num_tokens)) # stores tensor as non-trainable buffer that isn't updated during training
        self.enc = torch.nn.ModuleList([EncoderLayer(dim_hidden, dim_k) for _ in range(num_encoder_blocks)])
        self.classify = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_hidden),
            torch.nn.Linear(dim_hidden, output_shape)
        )

    def forward(self, x):
        # batch_size = x.shape[0]                            # x: [B, num_patch, dim_input]
        patch_emb = self.embedding(x)                      # [B, num_patch, dim_hidden]
        # create a class token
        # cls = self.cls.expand(batch_size, -1, -1)       # [B, 1, dim_hidden]
        # hdn = torch.cat([cls, pch], dim=1)                 # [B, num_tokens, dim_hidden]
        # hdn = hdn + self.position_emb(self.rng)         # [B, num_tokens, dim_hidden]
        patch_emb = patch_emb + self.position_emb(self.rng)
        # for enc in self.enc: hdn = enc(hdn)                # [B, num_tokens, dim_hidden]
        for enc in self.enc: patch_emb = enc(patch_emb)
        # cls_out = patch_emb[:, 0, :]                            # select only the 1st token, the class token [B, dim_hidden]
        # final = self.classify(cls_out)                          # [B, output_shape]

        # Aggregate over patches: e.g., take mean or sum over num_patches dimension
        pooled_output = patch_emb.mean(dim=1)  # (batch_size, dim_k)
        final = self.classify(pooled_output) # (batch_size, output_shape)
        return final
    