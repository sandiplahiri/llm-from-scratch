import torch
import torch.nn as nn

# Training deep neural networks with many layers can sometimes prove challenging due to problems like
# vanishing or explding gradients. These problems lead to unstable training dynamics and make it 
# difficult for the neural network to effectively adjust its weights, which means the learning process
# struggles to find a set of parameters (weights) for the neural network that minimizes the loss function. 
# In other words, the neural netwok has difficulty learning the underlying patterns in the data to 
# degree that would allow it to make accurate predictions or decisions.
#
# The main idea behind layer normalization is to adjust the activations (outputs) of a neural network
# layer to have a mean of 0 and a variance of 1, also known as unit variance. This adjustment speeds up
# the convergence to effective weights and ensures consistent, reliable training. 
#
# In GPT-2 and modern transformer architectures, layer normalization is typically aplied before and
# after the multi-head attention module and beforethe final output layer
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # eps is small constant that will be added to the variance to 
        # prevent division by zero during normaization
        self.eps = 1e-5
        # The scale and shift are two trainable parameters (of the same dimension as the input) that the 
        # LLM automatically adjusts during training if it is determined that doing so would improve the 
        # model's performance on its training task
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # This layer normalization operates on the last dimension of the input tensor x, 
        # which represent the embedding dimension (emb_dim).
        # Using keepdim=True in mean and variance calculations ensures that the output tensor
        # retains the same number of dimensions as the input tensor, even though the operation
        # operation reduces the tensor along the dimension specified via dim.  
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean)/torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift