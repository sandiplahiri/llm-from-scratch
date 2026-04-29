import torch
import tiktoken
from .gpt_model import GPTModel
from .llm_configs import GPT_CONFIG_124M


print("Creating sample dataset ...\n")
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

print("First dataset:\n", txt1)
print("Second dataset:\n", txt2)

print("Now tokenizing the dataset ...\n")

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print("Tokenized data set:\n", batch)

torch.manual_seed(123)
print("\nNow initializing the GPTModel using GPT-2 124M parametet model ...\n")

model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\n Output shape:", out.shape)
print(out)