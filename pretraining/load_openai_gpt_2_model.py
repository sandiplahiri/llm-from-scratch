from utils.gpt_download import download_and_load_gpt2
from utils.load_model_utils import load_weights_into_gpt
from utils.utils import generate,text_to_token_ids, token_ids_to_text
from .llm_configs import GPT_CONFIG_124M
from gpt_model import GPTModel
import torch
import tiktoken

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

model_configs = {
                  "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
                  "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
                  "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
                  "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
                }

# For example, load the smallest model: gpt2-small (124M)
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

# We can now use the updated NEW_CONFIG dictionary to initialize a new GPTModel instance
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

if torch.cuda.is_available():
    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
    # Use PyTorch 2.9 or newer for stable mps results
 #   major, minor = map(int, torch.__version__.split(".")[:2])
#   if (major, minor) >= (2, 9):
#        device = torch.device("mps")
else:
    device = torch.device("cpu")

load_weights_into_gpt(gpt, params)
gpt.to(device)

# Assuming the model is loaded correctly, let's generate new text using the generate function
torch.manual_seed(123)
tokenizer     = tiktoken.get_encoding("gpt2")
token_ids = generate(model=gpt,
                     idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
                     max_new_tokens=25,
                     context_size=NEW_CONFIG["context_length"],
                     top_k=50,
                     temperature=1.5)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))