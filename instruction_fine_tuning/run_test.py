from utils.utils import download_and_load_file, format_input, custom_collate_fn
import tiktoken


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
