import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    r"""Self-attention component of the transformer, which helps the large
    language model (LLM) to understand different patterns between
    words or tokens in a sequence.

    Args:
        dim_in (int):
            input dimension of the module.
        dim_out (int):
            output dimension of the module.
        context_length (int):
            Number of tokens that self-attention mechanism
            considers during inference.
        dropout_rate (float):
            probability of dropping weights during training.
            This helps to prevent overfitting, which can lead to exploding or
            vanishing gradient.
        num_heads (int):
            Number of attention heads. Each head looks at entire
            context and tries to identify the pattern within that context.
            The output of all heads are then concatenated and passed
            through the linear layer for the final result. The number of
            heads must be divisible of the output dimension.

            Increasing the number of attention heads can recognize more
            patterns,but comes at diminishing computational costs as
            more heads are added.
        qkv_bias (bool):
            Whether to add bias terms to multi-head attention
            parameters during training. These bias terms are known as
            Query (Q), Key (K), and Value (v). They are vectors that are
            used in the mechanism to help the model make extra adjustment
            to better learn from the data.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout_rate: float,
        num_heads: int,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert dim_out % num_heads == 0, \
            "output dimension must be divisible by num_heads"
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.Query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.output_projection = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout_rate)
        # Register buffers are used to ensure that the parameter is
        # non-trainable and will not be updated by gradient.
        # The values are saved along with the model
        self.register_buffer("upper_mask", torch.triu(
            torch.ones(context_length, context_length), diagonal=1))
