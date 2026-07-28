# rlx-text

Public surface for downstream LM apps:

- **Chat templates** — Jinja-rendered chat templates loaded inline or from
  GGUF metadata (`tokenizer.chat_template`, `tokenizer.ggml.bos_token_id`, …).
- **Tokenizer wrapper** — thin façade over the `tokenizers` crate so model
  crates don't all rebuild the same encode/decode glue.
- **Sampling** — re-exports `rlx_runtime::SampleOpts` plus helpers
  (`argmax`, top-k, top-p, repetition penalty).

Replaces the chat/sampling/tokenizer pieces that used to live in
`rlx-models/rlx-cli`, breaking the "runners depend on the CLI helper crate"
coupling.

## License

MIT OR Apache-2.0.
