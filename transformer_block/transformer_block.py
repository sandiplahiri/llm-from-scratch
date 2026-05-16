# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from .feed_forward_nn import FeedForward
from .layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.att = MultiHeadAttention(
                                      d_in=cfg["emb_dim"],
                                      d_out=cfg["emb_dim"],
                                      context_length=cfg["context_length"],
                                      num_heads=cfg["n_heads"],
                                      dropout=cfg["drop_rate"],
                                      qkv_bias=cfg["qkv_bias"]
                                     )
        
        self.ff            = FeedForward(cfg)
        self.norm1         = LayerNorm(cfg["emb_dim"])
        self.norm2         = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):

        # Shortcut connection for attention block
        shortcut = x

        # Step 1: Normalize
        x = self.norm1(x) 

        # Step 2: Multi-head self attention
        x = self.att(x)

        # Step 3: Dropout
        x = self.drop_shortcut(x)

        # Step 4: Add the shortcut connection
        x = x + shortcut

        # Shortcut connection for the feed forward block
        shortcut = x

        # Step 5: Normalize
        x = self.norm2(x)

        # Step 6: Feed forward network
        x = self.ff(x)

        # Step 7: Dropout
        x = self.drop_shortcut(x)

        # Step 8: Add the shortcut connection
        x = x + shortcut

        return x