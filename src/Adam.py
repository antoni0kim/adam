import torch
import torch.nn as nn

from src.adam_config import CONFIG_TYPE
from .components.Decoder import Decoder


class AdamModel(nn.Module):
    def __init__(self, CONFIG: CONFIG_TYPE):
        super().__init__()
        self.config: CONFIG_TYPE = CONFIG
        self.tok_emb = nn.Embedding(CONFIG["vocab_size"], CONFIG["emb_dim"])
        self.pos_emb = nn.Embedding(CONFIG["context_length"], CONFIG["emb_dim"])
        self.drop_emb = nn.Dropout(CONFIG["dropout_rate"])
        self.trf_blocks = nn.Sequential(
            *[Decoder(CONFIG) for _ in range(CONFIG["num_layers"])]
        )

        self.final_norm = nn.LayerNorm(CONFIG["emb_dim"])
        self.out_head = nn.Linear(CONFIG["emb_dim"], CONFIG["vocab_size"], bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, inputs):
        _, seq_len = inputs.shape
        if seq_len > self.config["context_length"]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.config['context_length']}"
            )
        tok_embds = self.tok_emb(inputs)
        pos_embds = self.pos_emb(torch.arange(seq_len, device=inputs.device))

        x = tok_embds + pos_embds.to(inputs.device)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
