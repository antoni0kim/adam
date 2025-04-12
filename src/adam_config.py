from typing import TypedDict


class CONFIG_TYPE(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float
    qkv_bias: bool


ADAM_CONFIG: CONFIG_TYPE = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "dropout_rate": 0.1,
    "qkv_bias": False,
}
