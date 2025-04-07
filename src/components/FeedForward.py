import torch.nn as nn

from src.adam_config import CONFIG_TYPE


class FeedForward(nn.Module):
    def __init__(self, CONFIG: CONFIG_TYPE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(CONFIG["emb_dim"], 4 * CONFIG["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * CONFIG["emb_dim"], CONFIG["emb_dim"]),
            nn.Dropout(CONFIG["dropout_rate"]),
        )

    def forward(self, inputs):
        return self.layers(inputs)
