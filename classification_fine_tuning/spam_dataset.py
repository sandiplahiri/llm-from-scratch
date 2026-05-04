import torch
import pandas as pd
from torch.utils.data import Dataset

# Loads data from the input csv file, tokenizes the text using the GPT 2 tokenizer from tiktoken,
# pads or truncates the sequences to a uniform length determined by either the longest sequence
# or a predefined maximum length. This ensures each input tensor is of the same size, which is
# necessary to create the batchs in the training data loader
class SpamDataset(Dataset):
    def __init__(self,
                 csv_file,
                 tokenizer,
                 max_length=None,
                 pad_token_id=50256):
        
        self.data = pd.read_csv(csv_file)

        # Pretokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [encoded_text[:self.max_length]
                                  for encoded_text in self.encoded_texts]
            
        # Pad sequences to the longest sequence
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
                              for encoded_text in self.encoded_texts]
        
    
    def __getitem__(self, index):

        encoded = self.encoded_texts[index]
        label   = self.data.iloc[index]["Label"]

        return (torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))
    
    def __len__(self):

        return len(self.data)
    
    def _longest_encoded_length(self):

        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length

        return max_length
