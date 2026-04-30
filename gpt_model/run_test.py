import torch
import tiktoken
from .gpt_model import GPTModel
from .llm_configs import GPT_CONFIG_124M
from utils.utils import generate_text_simple

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

# Calculate the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters: {total_params:,}")
# The number printed is 163,009,536. Although we initialized this model with a 
# 124 million-parametr GPT model, the original GPT-2 architecture reuses the weights from
# the token embedding layer in its output layer. This is called wegiht tying.
# So, let's look at the shapes of the token embedding layer and lineas output layer
# that we have initialized above
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# Let's remove the output layer parameter count from the total GPT-2 model count according to the weight tying
total_params_gpt2 = (
                      total_params - sum(p.numel() for p in model.out_head.parameters())
      )  
print(f"\nNumber of trainable parameters considering weight tying: {total_params_gpt2:,}")

# The above should print 124,412,160, which is the number of parameters in the original GPT-2 124M model

# Now run prediction
start_context = "Hello, I am"
print("Now running prediction on the sample dataset \"{start_context}\"...\n")
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
# Add batch dimension
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor_shape:", encoded_tensor.shape)

# Disable dropout since we are not training the model
model.eval()
out = generate_text_simple(model=model,
                           idx=encoded_tensor,
                           max_new_tokens=6,
                           context_size=GPT_CONFIG_124M["context_length"])
print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)