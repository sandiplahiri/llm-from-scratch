import torch
import tiktoken
from gpt_model import GPTModel
from .llm_configs import GPT_CONFIG_124M
from utils.utils import plot_losses, generate_text_simple, text_to_token_ids, token_ids_to_text, create_dataloader_v1, calc_loss_loader, train_model_simple


print("Test 1: GPTModel with modifiedGPT-2 124M parameter configuration as is (no pre-training) ...\n")

torch.manual_seed(123)
print("\nNow initializing the GPTModel using modified GPT-2 124M parameter model ...\n")
model = GPTModel(GPT_CONFIG_124M)
model.eval()

print("Creating sample dataset ...\n")
start_context = "Every effort moves you"
tokenizer     = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(model=model,
                                 idx=text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens=10,
                                 context_size=GPT_CONFIG_124M["context_length"])
print("Output text before pretraining:\n", token_ids_to_text(token_ids, tokenizer))

print("Test 2: Evaluating the model with the sample dataset before pretraining ...\n")
# Run this from the project directory using the command
# python -m utils.run_test.py
file_path = "data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Dataset details:")
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx   = int(train_ratio * total_characters)
train_data  = text_data[:split_idx]
val_data    = text_data[split_idx:]

train_loader = create_dataloader_v1(train_data,
                                    batch_size=2,
                                    max_length=GPT_CONFIG_124M["context_length"],
                                    stride=GPT_CONFIG_124M["context_length"],
                                    drop_last=True,
                                    shuffle=True,
                                    num_workers=0)

val_loader   = create_dataloader_v1(val_data,
                                    batch_size=2,
                                    max_length=GPT_CONFIG_124M["context_length"],
                                    stride=GPT_CONFIG_124M["context_length"],
                                    drop_last=False,
                                    shuffle=False,
                                    num_workers=0)

print("Train loader:")

for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:)")

for x, y in val_loader:
    print(x.shape, y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If this code is run on a machine a CUDA-supported GPU, the LLM will
# train on the GPU without making any changes to the code
model.to(device)

# Disable gradient tracking for efficiency because we are not training yet
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Traing loss:", train_loss)
print("Validation loss:", val_loss)

print("Test 3: Pre-training the model with the sample dataset  ...\n")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), # Returns all trainable weight parameters of the model
                              lr=0.0004,
                              weight_decay=0.1)

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(model,
                                                           train_loader,
                                                           val_loader,
                                                           optimizer,
                                                           device,
                                                           num_epochs=num_epochs, 
                                                           eval_freq=5,
                                                           eval_iter=5,
                                                           start_context="Every effort moves you",
                                                           tokenizer=tokenizer)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, 
            tokens_seen, 
            train_losses,
            val_losses)