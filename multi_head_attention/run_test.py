# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import torch
from multi_head_attention import MultiHeadAttention


# Instantiate an example text where each token is represented by a 3-dimensional embedding vector
inputs = torch.tensor(
                      [[0.43, 0.15, 0.89], # Your      (x^1)
                       [0.55, 0.87, 0.66], # journey]  (x^2)
                       [0.57, 0.85, 0.64], # starts    (x^3)
                       [0.22, 0.58, 0.33], # with      (x^4)
                       [0.77, 0.25, 0.10], # one       (x^5)
                       [0.05, 0.80, 0.55] # step       (x^6)
                     ]
                   )

# Create a two item batch by stacking the same inputs tensor
# How the dim parameter Affects Shape
# If you have N tensors, each of shape (A, B):
# a. dim=0: The new dimension is added at the beginning.
#    Resulting shape: (N, A, B)
# b. dim=1: The new dimension is added in the middle.
#    Resulting shape: (A, N, B) 
# c. dim=2 (or dim=-1): The new dimension is added at the end.
#    Resulting shape: (A, B, N) 
batch = torch.stack((inputs, inputs), dim=0)
# Expected output: (2, 6, 3 - batch_size, num_tokens, embedding_dim) 
print("Input batch shape:", batch.shape)   

# torch.manual_seed() in PyTorch sets a fixed seed for the random number generator (RNG) 
# across all devices (CPU and CUDA), ensuring reproducible results. 
# By initializing the RNG to a specific state, it guarantees that random operations—such 
# as weight initialization or data shuffling—produce the exact same sequence each time the script is executed.
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context vectors shape:", context_vecs.shape)
