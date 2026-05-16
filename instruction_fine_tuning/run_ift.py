# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

from utils.utils import download_and_load_file, format_input, custom_collate_fn, token_ids_to_text
from utils.instruction_dataset import InstructionDataset
import tiktoken
import torch
import json
import re
import time
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from utils.gpt_download import download_and_load_gpt2
from gpt_model import GPTModel
from utils.plot_utils import plot_losses
from utils.load_model_utils import load_weights_into_gpt
from utils.utils import (generate, 
                         text_to_token_ids, 
                         token_ids_to_text, 
                         calc_loss_loader, 
                         train_model_simple
                        )


# Download and load the training data 
file_path = "instruction-data.json"
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json")

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))

print("Example entry:\n", data[50])
print("Another example entry:\n", data[999])

model_input      = format_input(data[50])
desired_response = f"\n\n### Response: \n{data[50]['output']}"

print(model_input + desired_response)

# format_input() skips the Input: section is the 'input' field is empty
model_input    = format_input(data[999])
desired_response = f"\n\n### Response: \n{data[999]['output']}"

print(model_input + desired_response)

# Let's partition the dataset into training, test, and valiation sets
train_portion = int(len(data) * 0.85)
test_portion  = int(len(data) * 0.1)
val_portion   = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data  = data[train_portion:train_portion + test_portion]
val_data   = data[train_portion + test_portion:]

print("Traing set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))                   

tokenizer = tiktoken.get_encoding("gpt2")

# Instead of appending the <|endoftext|> tokens to the text inputs, we can append the
# token ID corresponding to <|endoftext|> to the rpretokenized inputs directly
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (inputs_1, inputs_2, inputs_3)
print(batch)

inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

# Let us setup device properly for the next steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#    device = torch.device("mps")
print("Device:", device)

# Create a partial function for the custom_collate_fn with device set to the current device
customized_collate_fn = partial(custom_collate_fn,
                                device=device,
                                allowed_max_length=1024)

# Now initialize the data loaders for the training, validation, and test sets
num_workers = 0
batch_size  = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader  = DataLoader(train_dataset,
                           batch_size=batch_size,
                           collate_fn=customized_collate_fn,
                           shuffle=True,
                           drop_last=True,
                           num_workers=num_workers)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader  = DataLoader(val_dataset,
                         batch_size=batch_size,
                         collate_fn=customized_collate_fn,
                         shuffle=True,
                         drop_last=True,
                         num_workers=num_workers)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader  = DataLoader(test_dataset,
                          batch_size=batch_size,
                          collate_fn=customized_collate_fn,
                          shuffle=True,
                          drop_last=True,
                          num_workers=num_workers)

print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

# Load the 355 million parameter GPT model as the base model for fine-tuning.
# The 124-million parameter model lacks the neccesary capacity to learn and
# retain the intricate patterns and nuanced behaviors required for high-quality
# instruction-following tasks
BASE_CONFIG = {"vocab_size":     50257, # Vocabulary size
               "context_length": 1024,  # Context length
               "drop_rate":      0.0,   # Dropout rate
               "qkv_bias":       True   # Query-key-value bias
               }

model_configs = {
                  "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
                  "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
                  "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
                  "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
                }

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(model_size=model_size,
                                          models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# Assess the pre-trained performance of this model
# Select an input data point
torch.manual_seed(123)
input_text = format_input(val_data[0])
print("\n\nInput text:\n", input_text)

# Generate the model's response to this input
token_ids = generate(model=model,
                     idx=text_to_token_ids(input_text, tokenizer),
                     max_new_tokens=35,
                     context_size=BASE_CONFIG["context_length"],
                     eos_id=50256)
generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = generated_text[len(input_text):].strip()
print("\n\nResponse text:\n", response_text)

# Calculate the initial loss for the training and validation sets
model.to(device)

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader,
                                  model,
                                  device,
                                  num_batches=5)
    
    val_loss   = calc_loss_loader(val_loader,
                                  model,
                                  device,
                                  num_batches=5)
    
print("\nTraining loss:", train_loss)
print("\nValidation loss:", val_loss)

# Now instruction fine-tune the model and evaluate it based on the first validation set instruction (val_data[0])
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=0.00005,
                              weight_decay=0.1)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(model,
                                                           train_loader,
                                                           val_loader,
                                                           optimizer,
                                                           device,
                                                           num_epochs=num_epochs,
                                                           eval_freq=5,
                                                           eval_iter=5,
                                                           start_context=format_input(val_data[0]), tokenizer=tokenizer)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes: .2f} minutes.")

# Plot the training and validation loss curves
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Now evaluate the fine-tuned model performance on the held-out test set
torch.manual_seed(123)

# Iterate over the first three test samples
test_num = 0
for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(model=model,
                         idx=text_to_token_ids(input_text, tokenizer).to(device),
                         max_new_tokens=256,
                         context_size=BASE_CONFIG["context_length"],
                         eos_id=50256)
    
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_num += 1
    print(f"Test {test_num}:\n")
    print(f"Input text:\n{input_text}")
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-" * 20)

# Use another LLM to evaluate this fine-tuned model's response. To prepare for this
# evaluation process, append the generated model responses to the test_set dictionary.
# Then save the updated updated data as a JSON file for record keeping. Additionally, 
# by saving this file, we can easily load and analyze the responses in separate
# Python sessions later on if needed. The following code uses the generate method in 
# the same manner as above. But it now iterates over the entire test set. Also, instead
# of printing the model responses, we add them to the test_set dictionary
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids  = generate(model=model,
                          idx=text_to_token_ids(input_text, tokenizer).to(device),
                          max_new_tokens=256,
                          context_size=BASE_CONFIG["context_length"],
                          eos_id=50256)
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text  = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)

print("n", test_data[0])
# Save the model as gpt-2medium355M-sft.pth to be able to reuse it in future projects
# Remove white spaces and parentheses from the file name
file_name = f"{re.sub(r'[ ()]', "", CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), file_name)

print(f"Model saved as {file_name}")

# The saved model can now be uploaded via model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
