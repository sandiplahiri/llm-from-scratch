# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import torch.nn as nn

# The feedforward network is a small neural network consisting of two Linear layers, and a GELU activation 
# function. The first layer increases the embedding dimension b by a factor of 4. The second layer decreases 
# the embedding dimension by a factor of 4, returning it to its original size.
#
# The FeedForward module plays a crucial role in enhancing the model's ability to learn from and generalize
# the data. Although the input and output dimensions of this module are the same, it internally expands the
# embedding dimension into a higher dimensional space through the first layer. This expansion is followed 
# by a nonlinear GELU activation and then a contraction back to the original dimension with the second  
# linear transformation. Such a design allows for the exploration of a richer representation state.
#
# Moreover, the uniformity in input and output dimensions simplifies the architecture by enabling the stacking
# of multiple layers without the need to adjust dimension between them, thus making the model more scalable
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                                    nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                                    nn.GELU(),
                                    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
                                   )
    
    def forward(self, x):
        return self.layers(x)