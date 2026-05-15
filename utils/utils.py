from requests import request
import torch
from torch.utils.data import DataLoader 

import tiktoken
from .data_loader import GPTDatasetV1

import json
import os
from tqdm import tqdm
import psutil
import urllib 

def download_and_load_file(file_path,
                           url):
     
     if not os.path.exists(file_path):
          with urllib.request.urlopen(url) as response:
               text_data = response.read().decode("utf-8")
          with open(file_path, "w", encoding="utf-8") as file:
               file.write(text_data)
     
     with open(file_path, "r") as file:
          data = json.load(file)
          
     return data

def format_input(entry):
     instruction_text = (
          f"Below is an instruction that describes a task. Write a response that appropriiately completes the request."
          f"\n\n### Instruction: \n{entry['instruction']}"  
     )

     input_text = f"\n\n### Input: \n{entry['input']}" if entry["input"] else ""

     return instruction_text + input_text


def custom_collate_fn(batch,
                      pad_token_id=50256,
                      ignore_index=-100,
                      allowed_max_length=None,
                      device="cpu"):
     
     batch_max_length = max(len(item) + 1 for item in batch)  # +1 for the end-of-sequence token
     inputs_1st, targets_1st = [], []

     for item in batch:
          new_item  = item.copy()
          new_item += [pad_token_id]

          # Pads sequences to max_length
          padded = (
               new_item + [pad_token_id] * (batch_max_length - len(new_item))
          )

          inputs  = torch.tensor(padded[:-1]) # Excludes the last token for inputs
          targets = torch.tensor(padded[1:]) # Shifts +1 to the right for targets

          # Replace all but the first padding tokens in targets by ignore_index
          # Get a boolean mask where True indicates padding tokens in targets
          mask    = targets == pad_token_id
          indices = torch.nonzero(mask).squeeze()

          if indices.numel() > 1:
               targets[indices[1:]] = ignore_index

          # Optionally truncate to the maximum sequence length
          if allowed_max_length is not None:
               inputs  = inputs[:allowed_max_length]
               targets = targets[:allowed_max_length]

          inputs_1st.append(inputs)
          targets_1st.append(targets)

     inputs_tensor = torch.stack(inputs_1st).to(device)
     targets_tensor = torch.stack(targets_1st).to(device)

     return inputs_tensor, targets_tensor

# Randomly splits the input dataset into training and validation datasets
def random_split(df,
                 train_frac,
                 validation_frac):
     
     # Shuffle the entire Dataframe
     df = df.sample(frac=1, random_state=123).reset_index(drop=True)
     
     # Calculate split indices
     train_end      = int(len(df) * train_frac)
     validation_end = train_end + int(len(df) * validation_frac)

     # Split the Dataframe
     train_df      = df[:train_end]
     validation_df = df[train_end: validation_end]
     test_df       = df[validation_end:]

     return train_df, validation_df, test_df
 

# Generates text for a GPT model by always picking the most probable next token
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

# Generates text for a GPT model using temperature and top-k sampling
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
    
    
# Converts text to token IDs
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

# Generates batches from input text
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

# Returns cross-entropy loss of a given batch given returned 
# via the training and validation loader
def calc_loss_batch(input_batch,
                    target_batch,
                    model,
                    device,
                    use_last_token=False):
       
       # Transfer to a GPU
       input_batch  = input_batch.to(device)
       target_batch = target_batch.to(device)
       
       if use_last_token == True:
            # Logits of the last output token
            logits = model(input_batch)[:, -1, :]
            loss   = torch.nn.functional.cross_entropy(logits, target_batch)
       else:
            logits = model(input_batch)
            loss   = torch.nn.functional.cross_entropy(logits.flatten(0, 1),    # Flattens (batch size, number of tokens, vocabulary size)
                                                                         # to (batch size * number of tokens, vocabulary size)
                                                        target_batch.flatten())
       
       return loss

# Computes the average training and validation loss over all batches
def calc_loss_loader(data_loader,
                     model,
                     device,
                     num_batches=None,
                     use_last_token=False):
     
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
             loss = calc_loss_batch(input_batch, target_batch, model, device, use_last_token)

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

# Fine-tunes the model to classify spam
def train_classifier_simple(model,
                            train_loader,
                            val_loader,
                            optimizer,
                            device,
                            num_epochs,
                            eval_freq,
                            eval_iter):
     
     # Initialize lists to track losses and examples seen
     train_losses, val_losses, train_accs, val_accs = [], [], [], []
     examples_seen, global_step = 0, -1

     for epoch in range(num_epochs):
          model.train()

          for input_batch, target_batch in train_loader:
               # Reset loss gradients from the previous batch iteration
               optimizer.zero_grad()

               loss = calc_loss_batch(input_batch,
                                      target_batch,
                                      model,
                                      device,
                                      use_last_token=True)
               
               # Calculate loss gradients
               loss.backward()

               # Update the model weights using loss gradients
               optimizer.step()

               # Track examples instead of tokens
               examples_seen += input_batch.shape[0]
               global_step += 1
               
               # Evaluate
               if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model,
                                                          train_loader,
                                                          val_loader,
                                                          device,
                                                          eval_iter,
                                                          use_last_token=True)
                    train_losses.append(val_loss)
                    val_losses.append(val_loss)

                    print(f"Ep {epoch+1} (Step {global_step:06d})): Train loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")
                
          # Calculate accuracy after each epoch
          train_accuracy = calc_accuracy_loader(train_loader,
                                                model,
                                                device,
                                                num_batches=eval_iter)
          
          val_accuracy   = calc_accuracy_loader(val_loader,
                                                model,
                                                device,
                                                num_batches=eval_iter)
          print(f"Training accuracy: {train_accuracy * 100: .2f}% | ", end="")
          print(f"Validation accuracy: {val_accuracy * 100: .2f}%")

          train_accs.append(train_accuracy)
          val_accs.append(val_accuracy)

     return train_losses, val_losses, train_accs, val_accs, examples_seen
     

# This function evaluates the input model
def evaluate_model(model,
                   train_loader,
                   val_loader,
                   device,
                   eval_iter,
                   use_last_token=False):
     
     # Dropout is disabled during evaluation for stable, reproducible results
     model.eval()

     # Disables gradient tracking, whcih is not requireds during evaluation,
     # to reduce the computational overhead
     with torch.no_grad():
          train_loss = calc_loss_loader(train_loader,
                                        model,
                                        device,
                                        num_batches=eval_iter,
                                        use_last_token=use_last_token)
          val_loss = calc_loss_loader(val_loader,
                                      model,
                                      device,
                                      num_batches=eval_iter,
                                      use_last_token=use_last_token)
          
          model.train()

          return train_loss, val_loss

# Queries a local model through REST API
def query_model(prompt,
                model="llama3",
                url="http://localhost:11434/api/chat"):
     
     # Create the data payload as a dictionary
     data = {"model": model,
             "messages": [{ "role": "user", 
                            "content": prompt }],
             "options": {"seed": 123, 
                         "temperature": 0,
                         "num_ctx": 2048}
             }
     
     # Converts the dictionary toa JSON-formatted string and encodes it to bytes
     payload = json.dumps(data).encode("utf-8")
     request = urllib.request.Request(url,
                                      data=payload,
                                      method="POST")
     
     # Create a request object, setting the method to POST and adding necessary headers
     request.add_header("Content-Type", "application/json")

     # Sends the request and captures the response
     response_data = ""
     with urllib.request.urlopen(request) as response:
          while True:
               line = response.readline().decode("utf-8")
               if not line:
                    break

               response_json  = json.loads(line)
               response_data += response_json["message"]["content"]

     return response_data

# Evaluate instruction fine-tuning LLM
def generate_model_scores(json_data,
                          json_key,
                          model="llama3"):
     
     scores = []
     for entry in tqdm(json_data, desc="Scoring entries"):
          prompt = (f"Given the input `{format_input(entry)}` "
                    f"and correct output `{entry['output']}`, "
                    f"score the model response `{entry[json_key]}`"
                    f" on a scale from 0 to 100, where 100 is the best score. "
                    f"Respond with the integer number only. "
                )
          
          score = query_model(prompt, model)
          try:
               scores.append(int(score))
          except ValueError:
               print(f"Could not convert score: {score}")
               continue
     
     return scores

# Checks whether the model improves during training.
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

# Calculates classification accuracy 
def calc_accuracy_loader(data_loader,
                         model,
                         device,
                         num_batches=None):
     
     model.eval()
     correct_predictions, num_examples = 0, 0

     if num_batches is None:
          num_batches = len(data_loader)
     else:
        # Reduce the number of batches to match the total number of batches in the data loader
          # if num_batches exceeds the number of batches in the data loader
          num_batches = min(num_batches, len(data_loader))

     for i, (input_batch, target_batch) in enumerate(data_loader):
          if i < num_batches:
               input_batch  = input_batch.to(device)
               target_batch = target_batch.to(device)

               with torch.no_grad():
                    # Logits of the last output token
                    logits = model(input_batch)[:, -1, :]

               predicted_labels = torch.argmax(logits, dim=-1)
               num_examples += predicted_labels.shape[0]
               correct_predictions += ((predicted_labels == target_batch).sum().item())
          else:
               break
          
     return correct_predictions / num_examples

# Verifies if a process name of "name" is running on the system
def check_if_running(process_name):

     running = False
     for proc in psutil.process_iter(["name"]):
          if process_name in proc.info["name"]:
               running = True 
               break
     
     return running