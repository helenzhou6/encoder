import torch
from torch import nn

class EncoderBlock(nn.Module):
    def __init__(self, dim_k, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_k, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_k)
        self.ff = nn.Sequential(
            nn.Linear(dim_k, dim_k),
            nn.ReLU(),
            nn.Linear(dim_k, dim_k)
        )
        self.norm2 = nn.LayerNorm(dim_k)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class MultiHeadEncoderModel(nn.Module):
    def __init__(self, num_classes, dim_k=49, num_heads=4, num_blocks=2, pos_embed=None):
        super().__init__()
        self.pos_embed_module = pos_embed
        self.encoder_blocks = nn.Sequential(*[
            EncoderBlock(dim_k, num_heads) for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(dim_k, num_classes)

    def forward(self, x):
        if self.pos_embed_module is not None:
            x = x + self.pos_embed_module()
        x = self.encoder_blocks(x)
        return x
        #pooled = x.mean(dim=1)
        #logits = self.classifier(pooled)
        #return logits