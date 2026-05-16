# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import json
from tqdm import tqdm
from utils.utils import (check_if_running,
                         query_model,
                         format_input,
                         generate_model_scores)


file_path = "instruction-data-with-response.json"
with open(file_path, "r") as file:
    test_data = json.load(file)



# Download, install and run 8-B parameterllama3 model using ollama
# Check if ollama is running
ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")

print("Ollama running:", check_if_running("ollama"))

# Check if the query_model function is working
model = "llama3"
result = query_model("What do Llamas eat?", model)
print("\nExample query: What do Llamas eat?")
print("\nResponse:", result)

# Apply this approach to the first three xamples form the test set 
for entry in test_data[:3]:
    prompt = (f"Given the input `{format_input(entry)}` "
              f"and correct output `{entry['output']}`, "
              f"score the model response `{entry['model_response']}"
              f" on a scale from 0 to 100, where 100 is the best score. "
             )
    
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n----------------------------------")

# Apply the generate_model_scores function to the entire test_data set
scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores): .2f}\n")