import requests
import zipfile
import os
from pathlib import Path
import pandas as pd
from utils.utils import random_split
from .spam_dataset import SpamDataset
import tiktoken

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