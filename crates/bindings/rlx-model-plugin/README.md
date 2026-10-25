# rlx-model-plugin: how to write a model plugin

A model plugin is a shared library with one or more model providers in it. rlx
loads it at run time, and then every language binding can open its models by
path or id with `Model.open(...)`. rlx never links the plugin, so a model
collection stays downstream of rlx. It depends on `rlx-runtime`, never the
other way round.

This crate is the reference plugin, with one small demo model per task:

- `demo:count` counts on from the last letter, so `chat("c")` gives `"defg…"`.
- `demo:voice` speaks letters, one sample each.
- `demo:speech` reads that audio back.
- `demo:embed` maps a text to `[length, vowels]`.

Each language's `models` example runs all four.

## Load one

```bash
cargo build -p rlx-model-plugin          # → target/debug/librlx_model_plugin.{dylib,so} / rlx_model_plugin.dll
RLX_MODEL_PLUGINS=target/debug/librlx_model_plugin.dylib python examples/models.py
```

- **`~/.rlx/plugins`** (or `$RLX_HOME/plugins`): copy a plugin library
  there and every program finds it. No setting is needed.
- **`RLX_MODEL_PLUGINS`** is a path list, separated by `:` (`;` on Windows),
  of libraries or directories of them. Both are read once, on the first model
  call.
- **`load_model_plugin(path)`** loads one explicitly, from any binding. It
  returns the names of the providers it registered.
- **`model_providers()`** lists every provider: the built-in `rlxp`, plus
  each plugin's.

## Link one, where nothing loads

iOS and the other Apple mobile systems, Android, wasm and consoles can't load
a plugin someone copied in. iOS loads only signed code shipped inside the app.
Android won't run code from an app's writable files. Wasm has no `dlopen`. On
these targets rlx doesn't look in `~/.rlx/plugins`. Instead the app links its
plugins into the library it ships, using the same plugin crate:

```toml
# Cargo.toml of a static library: librlx_capi with your models built in
[lib]
crate-type = ["staticlib"]

[dependencies]
rlx-capi    = { version = "…" }
rlx-runtime = { version = "…", default-features = false, features = ["linked-plugins"] }
my-models   = { version = "…" }
```

```rust
pub use rlx_capi::*;
rlx_runtime::link_model_plugins!(my_models);   // every plugin crate, once
```

The app calls `rlx_link_model_plugins()` once, before its first model call
(`linkModelPlugins()` in Swift). That call is what keeps the plugins in the
binary. A C linker drops the parts of a static library that nothing
references, and a provider registered at start-up is such a part.
`linked-plugins` also leaves out each plugin's `rlx_model_plugin_v1` table,
so several plugins can link into one library.

For an iOS app, `ios/build-xcframework.sh --capi-crate <your Cargo.toml>`
packages that library as the `RLX.xcframework` that `crates/bindings/rlx-swift`
builds against (`RLX_XCFRAMEWORK=1 RLX_CAPI_FEATURES=linked-plugins`).
[`rlx-capi/tests/linked`](../rlx-capi/tests/linked) is this crate, with the
reference plugin linked in. `cargo test -p rlx-capi --test c_program` links
`models.c` against it without LTO and checks every model.

A Rust host (an app or a game, or pyrlx or rlx-js embedded in an app)
depends on the plugin crate and calls `rlx_link_model_plugins()` the same way.
An app can also ship a plugin library and load it from its bundle with
`load_model_plugin(path)`: a signed `.framework` in an iOS app, or a `.so` in an
Android APK.

## Write one

A model is one trait for its task plus one `static` declaring the provider.
For a speech recogniser:

```rust
use rlx_runtime::model::{Claim, Provider, SpeechToText, speech_to_text_model};

struct Whisper { /* weights, compiled graphs, … */ }

impl SpeechToText for Whisper {
    fn family(&self) -> &str { "whisper" }
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, _opts: &serde_json::Value)
        -> anyhow::Result<String> { /* … */ }
}

static WHISPER: Provider = Provider::new("whisper", "speech-recognition", |req| {
    Ok(speech_to_text_model(Whisper::load(req.path(), req.device)?, req.model_info()))
})
.claims(&[Claim::GgufArch("whisper"), Claim::HfModelType("whisper")]);

rlx_runtime::export_model_plugin!("my-models", [WHISPER]);   // every provider, once
```

A list entry may carry `#[cfg(feature = "…")]`. Give the crate
`crate-type = ["cdylib", "rlib"]`, so it can be loaded and linked.

Every language then calls `Model.open("whisper-tiny.gguf").transcribe(audio, 16000)`.

| Task | Trait | Adapter | Callers use |
|---|---|---|---|
| `text-generation` | `LmRunner` + `TextCodec` | `lm_model` | `chat`, `generate_text`, `generate` |
| `speech-recognition` | `SpeechToText` | `speech_to_text_model` | `transcribe(audio, sample_rate)` |
| `text-to-speech` | `TextToSpeech` | `text_to_speech_model` | `synthesize(text)` → samples and their rate |
| `embedding` | `TextEmbedding` | `text_embedding_model` | `embed(texts)` → one vector per text |
| anything else | `Model` directly | — | `run` with named tensors |

**Claims** say which sources a provider opens. The best match across all
providers wins:

| Claim | Matches | Score |
|---|---|---|
| `Id("kokoro")` | exactly that id | 100 |
| `Prefix("kokoro:")` | ids that start with it | 90 |
| `GgufArch("qwen3")` | a GGUF file whose `general.architecture` is `qwen3` | 70 |
| `HfModelType("qwen3")` | a checkpoint whose `config.json` has `model_type: qwen3` | 70 |
| `FileName("model.onnx")` | a file or directory with that name | 50 |
| `Extension("onnx")` | a file with that extension | 10 |

For anything else, `.probe(|req| score)` adds a custom score.

- `req.model_info()` starts the model's info with the device and source
  filled in. `Provider` fills in the provider name and task.
- `req.option("voice")` reads a caller's option, by snake_case or camelCase.
- For chat, the codec's `apply_chat_template` turns `[{role, content}]`
  into the prompt. Without it, `chat` says the model has no template rather
  than guessing a format.
- A task method called on the wrong kind of model names the right one.
  `synthesize` on a speech recogniser says to use `transcribe`.

Loaded, the boundary is a C table, `rlx_model_plugin_v1`, at ABI version 1. Tensors,
JSON options and errors cross it as plain C data. That means the plugin's copy
of `rlx-runtime` doesn't have to be the host's; it only has to speak the same
ABI. A plugin built for another ABI is refused at load time with a message
saying so. `cargo test -p rlx-model-plugin` loads the exported table the way
`librlx_capi` does.

License: MIT OR Apache-2.0.
