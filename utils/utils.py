import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader 

import tiktoken
from .data_loader import GPTDatasetV1

# This function generates text for a GPT model by always picking the most probable next token
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

# This function generates text for a GPT model using temperature and top-k sampling
def generate(model,
             idx,  # idx is a (batch, n_tokens) array of indices in the current context
             max_new_tokens,
             context_size,
             temperature=0.0,
             top_k=None,
             eos_id=None):
    
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

        if top_k is not None:
             # Filter logits with top_k sampling
             top_logits, _ = torch.topk(logits, top_k)
             min_val       = top_logits[:, -1]
             logits        = torch.where(logits < min_val,
                                         torch.tensor(float('-inf')).to(logits.device),
                                         logits)
        
        if temperature > 0.0:
             logits   = logits / temperature
             probs    = torch.softmax(logits, dim=-1)
             idx_next = torch.multinomial(probs, num_samples=1)
        else:
             # Use greedy next token selection method when temperature scaling is diasbled
             # idx_next has shape (batch, 1)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            # Stop generating early if end-of-sequence token is encountered
            break

        idx = torch.cat((idx, idx_next), dim=-1)

    return idx
    
    
# This function converts text to token IDs
def text_to_token_ids(text, tokenizer):

    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})

    # .unsqueeze(0) adds the batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor

# This function converts token IDs to text
def token_ids_to_text(token_ids, tokenizer):

    # Removed batch dimension
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())

# This function generates batches from input text
def create_dataloader_v1(txt,
                         batch_size=4,
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0):
        
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,     # = True drops the last batch if it is shorter than 
                                                         # the specified batch_size to prevent loss spikes during training
                                num_workers=num_workers) # The number of CPU processes to use for preprocessing
        
        return dataloader

# This function returns cross-entropy loss of a given batch given returned 
# via the training and validation loader
def calc_loss_batch(input_batch,
                    target_batch,
                    model,
                    device):
       
       # Transfer to a GPU
       input_batch  = input_batch.to(device)
       target_batch = target_batch.to(device)
       
       logits = model(input_batch)
       loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1),    # Flattens (batch size, number of tokens, vocabulary size)
                                                                         # to (batch size * number of tokens, vocabulary size)
                                                target_batch.flatten())
       
       return loss

# This function computes the average training and validation loss over all batches
def calc_loss_loader(data_loader,
                     model,
                     device,
                     num_batches=None):
     
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
     
    if num_batches is None:
        # Iterate over all batches if no fixed num_batches is specified
        num_batches = len(data_loader)
    else:
         # Reduce the number of batches to match the total number of batches in the data loader
         # if num_batches exceeds the number of batches in the data loader
         num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
             loss = calc_loss_batch(input_batch, target_batch, model, device)

             # Sums loss for each batch
             total_loss += loss.item()
        else:
             break
    
    return total_loss / num_batches

# Main function to pre-train LLMs
def train_model_simple(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       device,
                       num_epochs,
                       eval_freq,
                       eval_iter,
                       start_context,
                       tokenizer):
     
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 1

    # Start the main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            # Reset loss gradients from the previous batch iteration
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,
                                   target_batch,
                                   model,
                                   device)
            # Calculate loss gradients
            loss.backward()

            # Update model weights using loss gradients
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model,
                                                      train_loader,
                                                      val_loader,
                                                      device,
                                                      eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} Step {global_step:06d}:"
                      f"Train loss {train_loss: .3f}, "
                      f"Val loss{val_loss: .3f}")
        
        generate_and_print_simple(model,
                                  tokenizer,
                                  device,
                                  start_context)
        
    return train_losses, val_losses, track_tokens_seen

# This function evaluates the input model
def evaluate_model(model,
                   train_loader,
                   val_loader,
                   device,
                   eval_iter):
     
     # Dropout is disabled during evaluation for stable, reproducible results
     model.eval()

     # Disables gradient tracking, whcih is not requireds during evaluation,
     # to reduce the computational overhead
     with torch.no_grad():
          train_loss = calc_loss_loader(train_loader,
                                        model,
                                        device,
                                        num_batches=eval_iter)
          val_loss = calc_loss_loader(val_loader,
                                      model,
                                      device,
                                      num_batches=eval_iter)
          
          model.train()

          return train_loss, val_loss
     
# This function helps to check whether the model improves during training.
# It takes a text snippet (start_context) as input, converts it into token IDs, 
# and feeds it to the LLM to generate a text sample using the generate_text_simple function
def generate_and_print_simple(model,
                              tokenizer,
                              device,
                              start_context):
     
     model.eval()
     context_size = model.pos_emb.weight.shape[0]
     encoded      = text_to_token_ids(start_context, tokenizer).to(device)

     with torch.no_grad():
          token_ids = generate_text_simple(model=model,
                                           idx=encoded,
                                           max_new_tokens=50,
                                           context_size=context_size)
    
     decoded_text = token_ids_to_text(token_ids, tokenizer)

     # Print in compact format
     print(decoded_text.replace("\n", " "))

     model.train()

# This function plots the training and validation set losses side by side
def plot_losses(epochs_seen,
                tokens_seen,
                train_losses,
                val_losses):
     
     fig, ax1 = plt.subplots(figsize=(5,3))
     ax1.plot(epochs_seen,
              train_losses,
              label="Training loss")
     ax1.plot(epochs_seen,
              val_losses,
              linestyle="-.",
              label="Validation loss")
     ax1.set_xlabel =("Epochs")
     ax1.set_ylabel("Loss")
     ax1.legend(loc="upper right")
     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
     ax2 = ax1.twiny()
     ax2.plot(tokens_seen, train_losses, alpha=0)
     ax2.set_xlabel("Tokens seen")
     fig.tight_layout()
     plt.show()

