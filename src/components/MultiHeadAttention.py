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

    Raises:
        ValueError: One of the input or output dimensions do not match the
            other.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout_rate: float = 0.0,
        num_heads: int = 1,
        qkv_bias: bool = False
    ):
        super().__init__()

        if dim_in != dim_out:
            raise ValueError(
                "`dim_out` must be divisible by `num_heads`."
            )

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.Query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.output_projection = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout_rate)
        # Register buffers are used to ensure that the parameter is
        # non-trainable and will not be updated by gradient.
        # The values are saved along with the model.
        # diagonal=1 here will not include diagonal and everything
        # below
        self.register_buffer("upper_mask", torch.triu(
            torch.ones(context_length, context_length), diagonal=1))

    def forward(self, inputs):
        batch, num_tokens, _ = inputs.shape
        keys = self.Key(inputs)
        queries = self.Query(inputs)
        values = self.Value(inputs)

        # reshape keys, queries, values to separate them into multiple
        # attention heads
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)

        # transpose (batch, num_tokens, self.num_heads, self.head_dim) to
        # (batch, self.num_heads, num_tokens, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #
        attention_scores = queries @ keys.transpose(2, 3)
        masks = self.upper_mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(masks, -torch.inf)

        # convert
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        #
        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(
            batch, num_tokens, self.dim_out)
        context_vector = self.output_projection(context_vector)

        return context_vector
