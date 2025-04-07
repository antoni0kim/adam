import torch.nn as nn

from .MultiHeadAttention import MultiHeadAttention
from .FeedForward import FeedForward

from src.adam_config import CONFIG_TYPE


class Decoder(nn.Module):
    def __init__(self, CONFIG: CONFIG_TYPE):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim_in=CONFIG["emb_dim"],
            dim_out=CONFIG["emb_dim"],
            context_length=CONFIG["context_length"],
            num_heads=CONFIG["num_heads"],
            dropout_rate=CONFIG["dropout_rate"],
            qkv_bias=CONFIG["qkv_bias"],
        )

        self.feed_forward = FeedForward(CONFIG)
        self.layer_norm1 = nn.LayerNorm(CONFIG["emb_dim"])
        self.layer_norm2 = nn.LayerNorm(CONFIG["emb_dim"])
        self.dropout = nn.Dropout(CONFIG["dropout_rate"])

    def forward(self, inputs):
        # Attention Block
        attn_out = self.attention(self.layer_norm1(inputs))
        x = inputs + self.dropout(attn_out)

        # Feed Forward block
        ff_out = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout(ff_out)

        return x
