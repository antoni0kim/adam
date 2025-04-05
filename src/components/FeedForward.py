import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, CONFIG):
        super().__init()
        self.layers = nn.Sequential(
            nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"]),
            nn.GELU(),
            nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"]),
        )

    def forward(self, inputs):
        return self.layers(inputs)
