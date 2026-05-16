# LLM From Scratch

Educational PyTorch implementation of GPT-style language model components, following a step-by-step path from attention layers to GPT-2 weight loading, pretraining, classification fine-tuning, and instruction fine-tuning.

The repository is organized as small runnable modules so each stage can be inspected independently before moving to the next one.

The code in this repo is from [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Dr. Sebastian Raschka and copyrighted by him.

## What Is Included

- Causal multi-head self-attention
- Transformer block with layer normalization and feed-forward network
- GPT-style model assembly
- GPT-2 tokenizer integration through `tiktoken`
- Simple text pretraining loop on `data/the-verdict.txt`
- OpenAI GPT-2 weight download and loading utilities
- Spam/ham classification fine-tuning
- Instruction fine-tuning and optional LLM-based evaluation through Ollama

## Repository Layout

```text
.
├── classification_fine_tuning/   # SMS spam classifier fine-tuning workflow
├── data/                         # The Verdict by Edith Wharton (1908) 
├── gpt_model/                    # GPT model definition and smoke test
├── instruction_fine_tuning/      # Instruction fine-tuning and evaluation scripts
├── multi_head_attention/         # Causal multi-head attention implementation
├── pretraining/                  # Pretraining and GPT-2 weight loading examples
├── transformer_block/            # LayerNorm, feed-forward, and transformer block code
├── utils/                        # Data loaders, training loops, generation, plotting, GPT download helpers
└── requirements.txt              # Python dependencies
```

Generated files such as model checkpoints, GPT-2 downloads, CSV splits, JSON outputs, plots, ZIP files, and virtual environments are ignored by Git.

## Requirements

- Python 3.10 or newer, but at least two versions older than the latest one
- PyTorch 2.2.2 or later, but older than 2.6


## Setup
- Install Python. 
  Download it from https://www.python.org/downloads/ and follow the installation directions. You can also use your operating system's standard command line tools. 
  &nbsp;
- Create a virtual environment
  - Install uv

    ```bash
    pip install uv
    ```

  - Create virtual environment

    ```bash
    uv venv --python=<Your installed Python version, e.g., python3.10 > 
    ``` 

- Activate this virtual environment

    ```bash
    source .venv/bin/activate
    ```

- Install the required packages
    ```bash
    uv pip install -r requirements.txt
    ```

## Quick Checks

Run the smaller component checks first:

```bash
python -m multi_head_attention.run_test
python -m transformer_block.run_test
python -m gpt_model.run_test
python -m utils.run_test
```

These scripts print intermediate tensors, shapes, parameter counts, and sample generation output.

## Pretraining

The pretraining example builds a GPT-style model, trains it on `data/the-verdict.txt`, plots losses, generates text, and saves a checkpoint under `model/`.

```bash
python -m pretraining.run_test
```

This script is more expensive than the component checks and may take time on CPU.

To load official GPT-2 weights and generate text:

```bash
python -m pretraining.load_openai_gpt_2_model
```

The GPT-2 weights are downloaded into `gpt2/`, which is ignored by Git.

## Classification Fine-Tuning

The classification workflow downloads the SMS Spam Collection dataset, balances ham/spam examples, creates train/validation/test CSV files, loads GPT-2 weights, replaces the output head with a binary classifier, fine-tunes selected layers, and saves `review_classifier.pth`.

```bash
python -m classification_fine_tuning.run_cft
```

This command needs network access the first time it downloads the dataset and GPT-2 weights.

## Instruction Fine-Tuning

The instruction fine-tuning workflow downloads an instruction dataset, loads GPT-2 medium weights, fine-tunes the model, writes generated test responses, and saves a fine-tuned checkpoint.

```bash
python -m instruction_fine_tuning.run_ift
```

This is the heaviest workflow in the repository. It downloads model/data files and can require substantial memory, disk space, and runtime.

After `run_ift` has produced `instruction-data-with-response.json`, optional LLM-based evaluation can be run with a local Ollama server and the `llama3` model available:

```bash
python -m instruction_fine_tuning.run_llm_based_eval
```

## Common Generated Artifacts

The following files and directories may be created while running scripts:

- `gpt2/`
- `model/`
- `sms_spam_collection/`
- `sms_spam_collection.zip`
- `train.csv`, `validation.csv`, `test.csv`
- `instruction-data.json`
- `instruction-data-with-response.json`
- `*.pth`
- `*.pdf`

They are intentionally ignored by Git because they are downloaded, generated, or large.

## Notes

- Run commands from the repository root so package imports and relative data paths resolve correctly.
- CUDA is used when available in most training scripts; otherwise they fall back to CPU.
- Some scripts are written as demonstrations and execute work at import time, so prefer running them as modules rather than importing them into other code.
