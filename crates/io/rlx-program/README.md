# rlx-program

RLX programs: a model's host logic (tokenizing, phonemizing, a search, a decode
loop) as a WebAssembly module that a `.rlxp` package carries beside its graphs
(an entry of kind `program`). rlx runs the module in a sandbox. The module runs
the package's functions through rlx, on any device, and does the rest itself.

This crate holds both sides of the ABI: the guest SDK the module is written
against, and the message format the runtime (`rlx-runtime`, feature `program`)
decodes.

```rust
use rlx_program::Message;

struct Echo;
impl rlx_program::Program for Echo {
    fn init(_config: serde_json::Value) -> Result<Self, String> { Ok(Echo) }
    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        let out = rlx_program::run("double", req.tensors)?;
        Ok(Message::tensors(out))
    }
}
rlx_program::program!(Echo);
```

Build it as a `cdylib` for `wasm32-unknown-unknown`.

## The ABI

| direction | call | what |
|---|---|---|
| module exports | `rlx_alloc(len) -> ptr`, `rlx_free(ptr, len)` | message buffers in the module's memory |
| module exports | `rlx_init(config)`, `rlx_call(task, request)` | each returns a response as `u64` (`ptr << 32 \| len`) that the caller frees |
| rlx provides | `run` / `run_with` / `run_sized` / `run_body` | run a package function, tensors in and out; `run_with` reads and keeps tensors in rlx's slots, so a decoder's caches never cross into the module |
| rlx provides | `host(service, message)` | the package's files and tokenizers, and the slots |
| rlx provides | `log(text)` | the program log (`RLX_PROGRAM_LOG`) |

A `Message` is a JSON body plus named tensors (`encode` / `decode`). An error is
a body `{"error": "…"}`. See [`docs/rlxp.md`](../../../docs/rlxp.md) for how a
package declares a program.

Only dependency: `serde_json`.

## License

MIT OR Apache-2.0.
