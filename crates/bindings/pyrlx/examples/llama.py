# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
"""LLaMA → RLX, exported in every form.  Run: `python llama.py`  (needs `transformers`)

The real HuggingFace `LlamaForCausalLM` architecture at a tiny config — rotary
embeddings, grouped-query attention, a computed causal mask, and SwiGLU. A real
checkpoint exports identically (weight names are HF-canonical); this just keeps
the graph small. int64 token ids are fed as f32 + a cast, so it runs on GPU too.
The `tensor` DSL can't express the computed-mask attention, so that form reports
n/a — use graph/flow (both cover every op).
"""

import torch
import torch.nn as nn

from _common import export_all_forms


def build():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, ids):
            return self.m(input_ids=ids).logits

    return Wrap().eval(), (torch.randint(0, 64, (1, 8)),)


if __name__ == "__main__":
    try:
        export_all_forms("llama", *build())
    except ImportError as e:
        print(f"llama example needs `transformers` ({e}); pip install transformers")
