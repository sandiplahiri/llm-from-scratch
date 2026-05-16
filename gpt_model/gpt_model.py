# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from transformer_block import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Create a list of TransformerBlock instances, repeating it cfg["n_layers"] times.
        # Then unpack the list using the * operator as separate arguments to nn.Sequential.
        # The result will be a nn.Sequential stacking them sequentially so data flows through each one in order
        self.trf_blocks = nn.Sequential(
                                        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
                                       )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    # This method takes a batch of input token indices, computes their embeddings, applies the positional 
    # embeddings, passes the sequence through the transform blocks, normalizes the final outpout, and 
    # then computes the logits, represening the next token's unnormalized probabilitiesfor each token in the vocabulary.
    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape

        # Step 1: Compute the token embeddings
        tok_embeds = self.tok_emb(in_idx)

        # Step 2: Apply the positional embeddings
        # The device setting will allow us to train the model on a CPU or GPU, depending on which device 
        # the input is sitting on
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Step 3: Add both embeddings together
        x = tok_embeds + pos_embeds

        # Step 4: Apply dropout to the combined embeddings
        x = self.drop_emb(x)

        # Step 5: Run it through the stack of transformer blocks
        x = self.trf_blocks(x)

        # Step 6: Apply the final layer normalization
        x = self.final_norm(x)

        # Step 7: Apply the output head to get the logits for each token in the vocabulary
        logits = self.out_head(x)

        return logits
