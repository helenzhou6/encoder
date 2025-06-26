import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(dim_model, dim_model * 4)
        self.linear2 = nn.Linear(dim_model * 4, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None):
        #tgt: Token embeddings (input so far).
	 	#memory: Output from the encoder.
		#tgt_mask: Causal mask (prevents looking ahead).
		#memory_mask: Optional mask over encoder outputs (rarely used here).
        # Self-attention with causal mask (residual + LayerNorm)
        _tgt = self.norm1(tgt + self._self_attention(tgt, tgt, tgt, tgt_mask, tgt_key_padding_mask))
        # Cross-attention to encoder output (memory)
        _tgt = self.norm2(_tgt + self._cross_attention(_tgt, memory, memory, memory_mask))
        # Feedforward
        _tgt = self.norm3(_tgt + self._feedforward(_tgt))
        return _tgt

    def _self_attention(self, query, key, value, attn_mask, tgt_key_padding_mask):
        attn_output, _ = self.self_attn(query, key, value, attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask)
        return self.dropout(attn_output)

    def _cross_attention(self, query, key, value, attn_mask):
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=attn_mask)
        return self.dropout(attn_output)

    def _feedforward(self, x):
        return self.dropout(self.linear2(F.relu(self.linear1(x))))

class DigitTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, dim_model=48, num_heads=2, num_layers=2, max_len=20):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim_model)
        self.pos_embed = nn.Embedding(max_len, dim_model)
        self.layers = nn.ModuleList([
            DecoderBlock(dim_model, num_heads)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(dim_model, vocab_size)

    def forward(self, tgt_seq, encoder_output, tgt_key_padding_mask=None):
        # tgt_seq: [batch_size, seq_len]
        positions = torch.arange(tgt_seq.size(1), device=tgt_seq.device).unsqueeze(0)
        x = self.token_embed(tgt_seq) + self.pos_embed(positions)

        # Causal mask (prevent attending to future tokens)
        seq_len = tgt_seq.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt_seq.device), diagonal=1).bool()

        # Padding mask: True where tgt_seq == PAD_TOKEN (usually 12), False elsewhere
        pad_token_id = 12  # or your defined PAD token constant
        padding_mask = tgt_seq == pad_token_id  # shape: [batch_size, seq_len]

        # Combine masks: A token should not attend to future tokens or to pads
        tgt_mask = causal_mask

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)

        return self.output_layer(x)

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image_patches, tgt_seq, tgt_key_padding_mask=None):
        # image_patches: shape [batch_size, num_patches, dim]
        # tgt_seq: shape [batch_size, seq_len]
        encoder_output = self.encoder(image_patches)
        logits = self.decoder(tgt_seq, encoder_output, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits
