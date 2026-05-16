# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

import torch

def classify_review(text,
                    model,
                    tokenizer,
                    device,
                    max_length=None,
                    pad_token_id=50256):
    
    model.eval()

    # PRepare input to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequence if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    # Add batch dimension
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        # Logit of the last output token
        logits = model(input_tensor)[:, -1, :]
    
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"