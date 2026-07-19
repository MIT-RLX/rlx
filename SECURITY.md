# Security Policy

## Supported versions

RLX is pre-1.0. Security fixes land on the latest `0.2.x` release; there is no
backport guarantee for older versions. Pin an exact version in production and
upgrade to pick up fixes.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public
issue for anything exploitable.

- Preferred: open a private security advisory via GitHub
  (**Security → Report a vulnerability**) on
  [`MIT-RLX/rlx`](https://github.com/MIT-RLX/rlx/security/advisories/new).

Include enough to reproduce: affected version/commit, platform and backend
(CPU / Metal / CUDA / …), a minimal graph or input, and the observed vs.
expected behavior. We aim to acknowledge within a few business days and will
coordinate a fix and disclosure timeline with you.

## Scope

RLX compiles and executes model graphs and loads model weight files (GGUF,
safetensors, ONNX). Reports involving memory-safety in kernels or loaders,
malformed-model-file handling, or unsafe FFI boundaries are in scope. Running
untrusted models or kernels is inherently sensitive — treat model files from
unknown sources as untrusted input.
