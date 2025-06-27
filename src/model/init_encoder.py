import torch
import torch.nn as nn

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
    def __init__(self, num_classes, dim_k=96, num_heads=4, num_blocks=6, image_size=96, patch_size=16):
        super().__init__()
        self.dim_k = dim_k
        self.patch_size = patch_size
        self.num_cuts = image_size // patch_size
        self.num_patches = self.num_cuts ** 2

        # --- PATCH EMBEDDING LAYERS ---
        self.linear_proj = nn.Linear(patch_size * patch_size, dim_k)
        self.row_embed = nn.Embedding(self.num_cuts, dim_k // 2)
        self.col_embed = nn.Embedding(self.num_cuts, dim_k // 2)

        # --- ENCODER BLOCKS ---
        self.encoder_blocks = nn.Sequential(*[
            EncoderBlock(dim_k, num_heads) for _ in range(num_blocks)
        ])

        # Optional classifier (not needed if using decoder)
        self.classifier = nn.Linear(dim_k, num_classes)

    def patch_image_tensor(self, img_tensor):
        patches = img_tensor.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(img_tensor.size(0), self.num_patches, -1)
        patch_embeddings = self.linear_proj(patches)

        # Positional encoding
        device = img_tensor.device
        positions = torch.arange(self.num_patches, device=device)
        rows = positions // self.num_cuts
        cols = positions % self.num_cuts
        pos_embed = torch.cat([self.row_embed(rows), self.col_embed(cols)], dim=1)
        patch_embeddings = patch_embeddings + pos_embed.unsqueeze(0)

        return patch_embeddings

    def forward(self, x):
        # Assume x is already patch-embedded
        x = self.encoder_blocks(x)
        return x