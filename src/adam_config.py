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
    "context_length": 256,  # 1024 original
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "dropout_rate": 0.1,
    "qkv_bias": False,
}
