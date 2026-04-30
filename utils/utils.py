import torch

def generate_text_simple(model,
                         idx,  # idx is a (batch, n_tokens) array of indices in the current context
                         max_new_tokens,
                         context_size):
    
    for _ in range(max_new_tokens):
        # Crops current context if it exceeds the context size. For example, if LLM
        # supports only 5 tokens, and the context size is 10, then only the 
        # *last* 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Focuses only on the last time step, so that (batch, n_tokens, vocab_size) becomes 
        # (batch, vocab_size). This is because we only want to predict the next token, which 
        # is based on the last token in the context
        logits = logits[:, -1, :]

        # probas has shape (batch, vocab_size), and it contains the probabilities of the 
        # next token being each token in the vocabulary
        probas = torch.softmax(logits, dim=-1)

        # idx_next has shape (batch, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Appends sampled indiex to the running sequence, where idx has shape (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=-1)

        return idx