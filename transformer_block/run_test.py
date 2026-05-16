# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import torch
from layer_norm import LayerNorm

torch.manual_seed(123)

# Create two training examples with five dimensions (features) each
batch_example = torch.randn(2,5)

print("Now testing LayerNorm ...\n")
ln     = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean   = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)
print("\nDone testing LayerNorm ...\n")