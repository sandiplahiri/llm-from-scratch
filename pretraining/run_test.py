import torch
import tiktoken
from gpt_model import GPTModel
from .llm_configs import GPT_CONFIG_124M
from utils.utils import generate_text_simple, text_to_token_ids, token_ids_to_text


print("Running test 1 for GPTModel with modifiedGPT-2 124M parameter configuration ...\n")

torch.manual_seed(123)
print("\nNow initializing the GPTModel using modified GPT-2 124M parametet model ...\n")
model = GPTModel(GPT_CONFIG_124M)
model.eval()

print("Creating sample dataset ...\n")
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(model=model,
                                 idx=text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens=10,
                                 context_size=GPT_CONFIG_124M["context_length"])
print("Output text before pretraining:\n", token_ids_to_text(token_ids, tokenizer))
