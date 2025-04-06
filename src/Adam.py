import torch
import torch.nn as nn

from src.adam_config import CONFIG_TYPE
from .components.Decoder import Decoder


class AdamModel(nn.Module):
    def __init__(self, CONFIG: CONFIG_TYPE):
        super().__init__()
        self.tok_emb = nn.Embedding(CONFIG["vocab_size"], CONFIG["emb_dim"])
        self.pos_emb = nn.Embedding(
            CONFIG["context_length"], CONFIG["emb_dim"])
        self.drop_emb = nn.Dropout(CONFIG["dropout_rate"])
        self.trf_blocks = nn.Sequential(
            *[Decoder(CONFIG) for _ in range(CONFIG["num_layers"])])

        self.final_norm = nn.LayerNorm(CONFIG["emb_dim"])
        self.out_head = nn.Linear(
            CONFIG["emb_dim"], CONFIG["vocab_size"], bias=False)

    def forward(self, inputs):
        _, seq_len = input.shape
        tok_embds = self.tok_emb(inputs)
        pos_embds = self.pos_emb(torch.arange(seq_len, device=inputs.device))

        x = tok_embds + pos_embds.to(inputs.device)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
