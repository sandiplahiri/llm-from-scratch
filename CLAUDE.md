# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Python virtual environment is at `.venv/`. Activate it before running anything:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Tests

Each module has a `run_test.py` that serves as a standalone demo/smoke test. Because `transformer_block` imports from `multi_head_attention` using an absolute package name, all tests must be run from the **project root** with the root on the Python path:

```bash
# Multi-head attention
PYTHONPATH=. python multi_head_attention/run_test.py

# Transformer block (LayerNorm only — does not exercise full TransformerBlock)
PYTHONPATH=. python transformer_block/run_test.py

# Full GPT model end-to-end
PYTHONPATH=. python gpt_model/run_test.py
```

## Architecture

The project builds a GPT-2 style LLM bottom-up, with each directory being a self-contained Python package:

```
multi_head_attention/   → MultiHeadAttention (causal, multi-head, with out-projection)
transformer_block/      → LayerNorm, FeedForward, TransformerBlock (pre-norm + residuals)
gpt_model/              → GPTModel (token + positional embeddings → transformer stack → logits)
                          llm_configs.py holds GPT_CONFIG_124M (GPT-2 small hyperparameters)
```

**Data flow through GPTModel:**
1. Token indices → `tok_emb` + `pos_emb` (learned embeddings summed)
2. Dropout on combined embeddings
3. `n_layers` stacked `TransformerBlock`s (each: pre-norm → MHA → residual → pre-norm → FFN → residual)
4. Final `LayerNorm`
5. Linear `out_head` → logits over vocabulary

**TransformerBlock** follows the pre-norm (GPT-2) convention: `LayerNorm` is applied *before* each sub-layer, and the residual shortcut is added *after*.

**Cross-module imports:** `transformer_block/transformer_block.py` imports `MultiHeadAttention` from the `multi_head_attention` package, and `gpt_model/gpt_model.py` imports `TransformerBlock` and `LayerNorm` from `transformer_block`. These are absolute package imports, so the project root must be on `sys.path` (hence `PYTHONPATH=.`).

## Known Issues

- `feed_forward_nn.py:25` — typo: `self.layes(x)` should be `self.layers(x)`
- `gpt_model.py:37` — typo: `devices=in_idx.device` should be `device=in_idx.device`
- `llm_configs.py` uses key `"n_heads"` but `TransformerBlock.__init__` reads `cfg["num_heads"]` — these must be kept in sync when passing configs
