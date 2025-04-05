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
            qkv_bias=CONFIG["qkv_bias"]
        )

        self.feed_forward = FeedForward(CONFIG)
        self.layer_norm1 = nn.LayerNorm(CONFIG["emb_dim"])
        self.layer_norm2 = nn.LayerNorm(CONFIG["emb_dim"])
        self.shortcut_dropout = nn.Dropout(CONFIG["dropout_rate"])

    def forward(self, inputs):
        shortcut = inputs
        inner_path = self.layer_norm1(inputs)
        inner_path = self.attention(inner_path)
        inner_path = self.shortcut_dropout(inner_path)
        inner_path = inner_path + shortcut

        shortcut = inner_path
        inner_path = self.layer_norm2(inner_path)
        inner_path = self.feed_forward(inner_path)
        inner_path = self.shortcut_dropout(inner_path)
        inner_path = inner_path + shortcut

        return inner_path
