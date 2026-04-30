import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_in, 
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.W_query   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key     = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj  = nn.Linear(d_out, d_out)
        self.dropout   = nn.Dropout(dropout)

        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), diagonal=1)
        ) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        # Reshapes the keys, queries, and values to have the shape (batch_size, num_tokens, num_heads, head_dim)
        # Implicitly splits the matrix by adding a num_heads dimension.
        # Then unrolls the last dim: (b, num_tokens, d_out) to (b, num_tokens, num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose from 
        # shape(b, num_tokens, num_heads, head_dim) to 
        # shape(b, num_heads, num_tokens, head_dim)
        # This transposition is crucial for correctly aligning the queries, keys, and values 
        # across difference heads and performing batched matrixed multilplications effectively
        keys    = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # Computes the dot product for each head
        # The matrix multiplication implementation of PyTorch handles the found-dimensional input tensor
        # so that the matrix multiplication is carried out between two last dimensions (num_tokenn, head_dim)
        # and then repeated for the individual heads
        attn_scores = queries @ keys.transpose(2,3)

        # Applies causal masking by setting the upper triangular part of attn_scores to -inf
        # Truncates the masks of size context_length x context_length to the
        # number of tokens in the input sequence, i.e., to num_tokens x num_tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Uses the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalizes the attention scores using softmax
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)

        # Applies dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # Computes the weighted sum of the values using the attention weights
        # Tensor shape: (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values
         # Tensor shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2) 

        # Combines heads, where self_d_out = self.num_heads * self_head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        print("[MultiHeadAttention] context_vec before optional projection:", context_vec)
        print("[MultiHeadAttention] context_vec shape before optional projection:", context_vec.shape)


        # Adds an optional linear projection
        context_vec = self.out_proj(context_vec)
        print("[MultiHeadAttention] context_vec after optional projection:", context_vec)
        print("[MultiHeadAttention] context_vec shape after optional projection:", context_vec.shape)

        return context_vec
