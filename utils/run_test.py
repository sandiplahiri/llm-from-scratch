# Copyright (c) Sebastian Raschka.
# Source code: "Build a Large Language Model From Scratch"

from .utils import create_dataloader_v1

# Run this from the project directory using the command
# python -m utils.run_test.py
# Test create_dataloader_v1
print("Now testing create_dataloader_v1 ...")
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text,
                                  batch_size=1,
                                  max_length=4,
                                  stride=1,
                                  shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)