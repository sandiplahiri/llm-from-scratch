# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import requests
import zipfile
import os
from pathlib import Path
import pandas as pd
from utils.utils import random_split
from .spam_dataset import SpamDataset
import tiktoken
import torch
from torch.utils.data import DataLoader
from pretraining.llm_configs import GPT_CONFIG_124M
from utils.gpt_download import download_and_load_gpt2
from gpt_model import GPTModel
from utils.load_model_utils import load_weights_into_gpt
from utils.utils import train_classifier_simple, calc_loss_loader, calc_accuracy_loader, generate, generate_text_simple, text_to_token_ids, token_ids_to_text
from utils.plot_utils import plot_values
import time
from .classify_review import classify_review


# Downloads and unzips the spam/no-spam dataset
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


# Creates a balanced dataset for spam/no-spam data
def create_balanced_dataset(df):

    num_spam   = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df



# Download the spam/no-spam dataset 
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

try:
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
except (requests.exceptions.RequestException, TimeoutError) as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(data_file_path,
                 sep="\t",
                 header=None,
                 names=["Label", "Text"])

print(df)
print(df["Label"].value_counts())

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

# Convert the string class labels "ham" and "spam" into integer class labels 0 and 1, respectively
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# Save the dataset as CSV files for later reuse
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="train.csv",
                            max_length=None,
                            tokenizer=tokenizer)
print("The longest in training dataset sequence:", train_dataset.max_length)
print("train_dataset[0]:", train_dataset[0])

# Pad the validation and test sets to match the length of the longest training sequence.
# Any validation and test samples exceeding the length of the longest training example 
# are truncated using the encdoded_text[:self.maximum_length] in the SpamDataset class.
# THis truncation is optional, we can set max_length=None for both validation and test sets, 
# provided there are no sequences exceeding 1024 tokens in these sets. Note that the model
# can handle sequences up to 1024 tokens, given its context length limit
val_dataset = SpamDataset(csv_file="validation.csv",
                          max_length=train_dataset.max_length,
                          tokenizer=tokenizer)

test_dataset = SpamDataset(csv_file="test.csv",
                           max_length=train_dataset.max_length,
                           tokenizer=tokenizer)

# Create PyTorch data loaders
num_workers = 0
batch_size  = 8
torch.manual_seed(123)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True)

val_loader   = DataLoader(dataset=val_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True)

test_loader  = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=True)

for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions:", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# Next, we initiliaze a model with pretrained weights
CHOOSE_MODEL = "gpt2-small (124M)"

model_configs = {
                  "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
                  "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
                  "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
                  "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
                }

BASE_CONFIG = GPT_CONFIG_124M.copy()
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])  
BASE_CONFIG.update({"context_length": 1024})
BASE_CONFIG.update({"qkv_bias": True})

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text_1 = "Every effort moves you"
token_ids = generate_text_simple(model=model,
                                 idx=text_to_token_ids(text_1, tokenizer),
                                 max_new_tokens=15,
                                 context_size=BASE_CONFIG["context_length"])

print(token_ids_to_text(token_ids, tokenizer))

print(model)

# For classification fine-tuning, freeze the model, i.e., make all layers nontrainable 
for param in model.parameters():
    param.requires_grad = False

# Then, replace the output layer (model.out_head), which originally maps the layer inputs
# to 50,257 dimensions, the size of the vocabulary
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"],
                                 out_features=num_classes)

# Fine-tuning additional layets can noticeable improve the predictive 
# performance of the model. We also configure the last transformer block
# and the final LayerNorm module, which connects this block to the output
# layer, to be trainable.
# To make the final LayerNorm and last transformer block trainable, set
# their respective requires_grad to True
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

# At the output layer, we will get a [1, num_tokens/context_length, 2].
# As we are interested in fine-tuning this model to return a class label
# indicating whether a model output is "spam" or "not spam", we do not 
# need to fine-tune all num_token output rows. Instead, we can focus on a single 
# output token. In particular, we will focus on the last row corresponding to the 
# last output token.
# Given the causal attention mask setup, the last last token in a sequence 
# accumulates the lost information since it is the only token with access
# to data from all the previous tokens. Therefore, in this spam classification 
# task, we focus on this last token during the fine-tuning process

# Calculate classificationa accuracies across various datasets for 10 batches
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader,
                                      model,
                                      device,
                                      num_batches=10)

val_accuracy   = calc_accuracy_loader(val_loader,
                                      model,
                                      device,
                                      num_batches=10)

test_accuracy  = calc_accuracy_loader(test_loader,
                                      model,
                                      device,
                                      num_batches=10)

print(f"\nTraining accuracy: {train_accuracy*100: .2f}%")
print(f"Validation accuracy: {val_accuracy*100: .2f}%")
print(f"Test accuracy: {test_accuracy*100: .2f}%")

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader,
                                   model,
                                   device,
                                   num_batches=5,
                                   use_last_token=True)
    
    val_loss   = calc_loss_loader(val_loader,
                                   model,
                                   device,
                                   num_batches=5,
                                   use_last_token=True)
    
    test_loss  = calc_loss_loader(test_loader,
                                   model,
                                   device,
                                   num_batches=5,
                                   use_last_token=True)
    
print(f"Training loss: {train_loss: .3f}%")
print(f"Validation loss: {val_loss: .3f}%")
print(f"Test loss: {test_loss: .3f}%")

start_time=time.time()
torch.manual_seed(123)

optimizer  = torch.optim.AdamW(model.parameters(),
                               lr=5e-5,
                               weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(model,
                                                                                        train_loader,
                                                                                        val_loader,
                                                                                        optimizer,
                                                                                        device,
                                                                                        num_epochs=num_epochs,
                                                                                        eval_freq=50,
                                                                                        eval_iter=5)

end_time = time.time()

execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes: .2f} minutes.")

# Plot the resulting loss curves
epochs_tensor = torch.linspace(0,
                                num_epochs,
                                len(train_losses))

examples_seen_tensor = torch.linspace(0,
                                      examples_seen,
                                      len(train_losses))

plot_values(epochs_tensor,
            examples_seen_tensor,
            train_losses,
            val_losses)

# Plot the classification accuracies
epochs_tensor = torch.linspace(0,
                                num_epochs,
                                len(train_accs))

examples_seen_tensor = torch.linspace(0,
                                      examples_seen,
                                      len(train_accs))

plot_values(epochs_tensor,
            examples_seen_tensor,
            train_accs,
            val_accs,
            label="accuracy")

# Calculate the performance metrics for training, validation, and test sets 
# across tjhe entire dataset, this time without definiting the iter_eval value
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy * 100: .2f}%")
print(f"Validation accuracy: {val_accuracy * 100: .2f}%")
print(f"Test accuracy: {test_accuracy * 100: .2f}%")

text_1 = ("You are a winner you have been specially selected to receive $1000 cash or a $2000 award")

print(classify_review(text_1, 
                      model,
                      tokenizer,
                      device,
                      max_length=train_dataset.max_length))

text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
print(classify_review(text_2, 
                      model,
                      tokenizer,
                      device,
                      max_length=train_dataset.max_length))

# Save the model
torch.save(model.state_dict(), "review_classifier.pth")

# Load the model
model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)