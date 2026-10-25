# rlx-web

WebAssembly entry point for RLX — run models in the browser.

It compiles to `wasm32-unknown-unknown` and exposes a small JS-callable surface
via `wasm-bindgen`.

## Real transformer in the browser

`transformer_next_logits(tokens, vocab, dim, n_layers, n_heads, head_dim, ffn,
seed)` runs a **real decoder-only transformer** — RMSNorm + RoPE + causal
multi-head attention + SwiGLU MLP — on the CPU backend and returns the
next-token logits. Weights are **deterministically synthesized** from `seed`, so
this is a faithful *architecture* run (real graph, real ops), not pretrained
weights. It's verified end-to-end natively (`src/transformer.rs` tests):
determinism, output shape, and the **causal property** (a position's logits
never depend on later tokens), plus isolations for bare attention and
attention+RoPE.

Loading real GGUF pretrained weights is a separate, still-missing piece: the
old `rlx-models` GGUF→Qwen3 runner was deleted and has no in-tree replacement,
and the byte-level GGUF parser (`rlx_gguf::GgufFile::from_reader`) + an
HF↔GGUF name/transpose mapping would need rebuilding to feed real weights into
this graph.

### Two things that had to be fixed for *any* model to run in a browser
- **BLAS:** `rlx-cpu`'s default `blas` feature declared an `extern "C"
  cblas_sgemm` that becomes an unsatisfiable wasm import (instantiate
  `LinkError`). It's now routed to the pure-Rust scalar GEMM on wasm
  (`blas.rs` cfgs gated `not(target_arch="wasm32")`).
- **32-bit overflow:** `rlx-runtime`'s decode memory-budget constants
  (12 GiB / 4 GiB) overflowed 32-bit `usize`; computed in `u64` now.

### CPU fusion bug surfaced here — now fixed
Building this transformer surfaced a real backend bug: the CPU attention-block
fusion (`rlx-cpu`, `Thunk::FusedAttnBlock`) collapsed
`QKV → narrow×3 → RoPE×2 → attention → out-proj` into one kernel but **dropped
the mask kind**, applying only a per-key padding mask. A `MaskKind::Causal`
decoder therefore attended to future tokens (the 1-layer leak was ~0.07;
`skip_fusion` gave 0). The fix threads `mask_kind` through the fused block and
synthesizes the causal / sliding-window mask in-kernel, so `transformer_logits`
now compiles with the **default** fusion pipeline. Guarded by
`rlx-cpu`'s `fused_attn_block_respects_causal_mask` and this crate's
`fused_path_is_causal_like_unfused` / `causal_*` tests.

## Quickstart

```sh
# one command, all platforms — builds the bundle and serves the demo
just serve-web

# same, but also bring up a WebGPU device
just serve-web --webgpu
```

Then open <http://localhost:8000>. (ES modules + wasm must be served over
`http://`, not opened as `file://`.)

Prerequisite: the `wasm-bindgen` **CLI** must match the `wasm-bindgen` crate
version. The build script checks this and prints the exact command if not:

```sh
cargo install wasm-bindgen-cli --version <crate-version>
```

Build without serving: `just build-web` (add `--webgpu`, `--webgl`, or `--all`).
The bundle lands in `crates/bindings/rlx-web/web/pkg/`.

## Compute paths

Every path runs **both forward and backward** (the backward pass is the
[`rlx_autodiff`] gradient graph). The graphs are identical across backends —
only the executor differs.

| Path | Mechanism | Sync? | Verification |
|------|-----------|-------|--------------|
| **CPU** (default) | `rlx_runtime::Session` on `rlx-cpu` (single-threaded on wasm) | sync | ✅ native test: grads match finite-differences; SGD reduces loss |
| **WebGPU** (`--webgpu`) | compute shaders via `rlx_wgpu::WgpuExecutable::run_async` | async (GPU→CPU `map_async` can't block the event loop) | native: wgpu tests; in-browser: `just serve-web --webgpu` |
| **WebGL2** (`--webgl`) | render-to-texture via `rlx-webgl` (no compute shaders on WebGL2) | sync (`readPixels`) | ✅ native: planner+numerics vs autodiff; in-browser: `just serve-web --webgl` |

The GPU executors can only be *run* in a browser, so they are compile-verified
here and validated by serving the demo, which cross-checks GPU results against
the CPU reference. The WebGL backend's planner + numerics are additionally
verified natively (see `rlx-webgl`).

## Vision benchmark (browser)

Four classification models run in the browser via [`VisionBench`](src/api.rs):

| Slug | Architecture |
|------|----------------|
| `mnist-cnn` | TinyConv MNIST (matches rlx-cortexm trainer) |
| `mnist-mlp` | Flattened 784→128→10 MLP |
| `cifar-cnn` | 3-block CNN for 32×32 RGB |
| `resnet` | CIFAR-sized ResNet-style (two residual blocks) |

Open <http://localhost:8000/vision-bench.html> after `just serve-web --all`.

## Tab cluster — one data-parallel rank per browser tab

Open <http://localhost:8000/cluster.html> after `just serve-web`, then open it
again in another tab. Each tab is a **rank**: it compiles the backward graph
once, computes gradients over *its own shard* of the dataset, and the cohort
averages them every step. Samples/s rises with the tab count; the loss curve is
the cohort's.

```text
  tab 1                         tab 2
  ┌──────────────┐              ┌──────────────┐
  │ UI thread    │              │ UI thread    │   ← renders state only
  │   ▲ state    │              │   ▲ state    │
  │ Worker       │              │ Worker       │
  │  ClusterWorker (wasm)       │  ClusterWorker (wasm)
  └──────┬───────┘              └──────┬───────┘
         └────── BroadcastChannel ─────┘            ← gradients + parameters
```

Compute and coordination both live in a dedicated Worker, so gradient buckets
never touch a UI thread and the page stays responsive while training runs.

### Backends, per rank

Each rank picks `auto` / `webgpu` / `webgl` / `cpu`, and the roster shows what
every peer **actually resolved to** — never what it was asked for. What can
train where is narrower than what compiles:

| Model | CPU | WebGPU | WebGL2 |
|-------|-----|--------|--------|
| `mnist-mlp` | ✅ | ✅ | ✅ |
| `mnist-cnn`, `cifar-cnn`, `resnet` | ✅ | ❌ host step | ❌ no conv backward |

Both GPU refusals are real and neither is a compile error:

- **WebGL2** has no conv backward in this stack
  ([`VisionModelId::webgl_train`]), so it trains the MLP only. It runs in the
  Worker because `rlx-webgl` gets its context from an `OffscreenCanvas`, not
  from `document`.
- **WebGPU** refuses a conv graph, and it is the **backward** pass that does
  it, not the convolution itself: forward conv is native, while
  `conv2d_backward_input_host` and `conv2d_backward_weight_host` are host steps,
  and the browser WebGPU path refuses any schedule with host steps because it
  cannot block to run them. Measured per model by
  `tests/webgpu_trainable_models.rs`, which compiles each training graph for
  wgpu and reports `CompiledGraph::host_execution_summary()`:

  | Model | On WebGPU |
  |-------|-----------|
  | `mnist-mlp` | entirely on device |
  | `mnist-cnn` | 3 of 31 steps on the host, 2 sync points |
  | `cifar-cnn` | 5 of 42 steps on the host, 3 sync points |
  | `resnet` | 12 of 65 steps on the host, 6 sync points |

  So a conv rank resolving to CPU is correct, not a misconfiguration — and
  `ClusterWorker::fallback` now carries the reason to the UI, because "why is it
  only computing on CPU?" previously had no answer anywhere on the page. The
  page defaults to `mnist-mlp` for that reason, and the model picker labels the
  rest `cpu only`.

So backend selection is two-stage. [`backend_candidates`] is pure and
model-aware — it returns every viable backend best-first, and skips WebGL for a
model it cannot train, which matters because `auto` otherwise resolves by device
priority alone (`WebGpu`, `OpenGl`, `Cpu`) and would hand a conv model to WebGL
on a browser with no WebGPU. The rank then walks that list, compiling each and
rejecting any whose schedule needs host execution. That last check needs
[`BrowserCompiledGraph::requires_host_execution`], added for this: the flag
existed on the wgpu executable but nothing outside `run_async` could ask, so the
failure could only be discovered one step at a time.

An explicit request yields exactly one candidate and is refused with a reason
rather than silently running elsewhere; `auto` falls through to CPU.

### GitHub Pages: every page

`.github/workflows/pages.yml` publishes all three pages — the forward/backward
demo at the root, `vision-bench.html` and `cluster.html` — over one wasm bundle
built with both browser GPU paths (`build.py --all`). It runs on pushes to
`main` that touch `crates/`, `Cargo.toml` or `Cargo.lock`, and on demand. Turn
it on once under **Settings → Pages → Source: GitHub Actions**.

The site also carries the **Qwen3.5 chat** (`qwen/chat.html`) from
[rlx-models](https://github.com/MIT-RLX/rlx-models)' `rlx-models-web`: the
workflow checks that repo out next to this one, patches every `rlx*` crate to
this checkout and builds it with `wasm-pack` (WebGPU + WebGL2). Pages serves only
the page and its 12 MB wasm; the GGUF streams from the Hugging Face Hub (Range
requests, each worker fetching its own tensors), so no weights live in the site.
`--qwen DIR` adds it and links it from the pages' nav. A push to rlx-models does
not redeploy; run the workflow by hand for that.

    just build-pages                        # the same site, in ./_site
    just build-pages _site ../rlx-models/crates/rlx-models-web/web   # + the Qwen chat
    python3 -m http.server -d _site 8000

The site is assembled by the builder below with `--all-pages`, which keeps each
page under its own name (they link each other) and runs the same
missing-reference check.

### Deploying it as a standalone static site

    just build-cluster-site                 # → examples/tab-cluster/
    python3 crates/bindings/rlx-web/tools/build_site.py --out /tmp/site --open

The result is self-contained and needs no backend: 10 files, ~8.9 MB (the wasm
bundle is 8.4 MB of it), with the demo as `index.html` and a `.nojekyll` so
GitHub Pages does not skip `pkg/`.

Files are **discovered, not listed**. The builder walks the page's real
dependency graph — `<script src>`, `<img src>`, `import … from "./x.js"`, and
`new Worker("./x.js")` — so adding a module needs no manifest edit here. That
last pattern is in the list because it was missing first time: a worker
constructor is not an import, so the first build shipped a site that loaded
cleanly and then died creating its worker. The builder now **verifies its own
output**, re-walking the copied files and failing if any local reference does
not resolve, which catches the next missed pattern without knowing what it is.

Verified by serving the assembled directory with `python3 -m http.server` and
running two browsers against it: cohort of 2 over the WebRTC mesh, 3,443
samples/s, masthead and highlighting intact.

The generated `README.md` carries a **Hugging Face Space card**, so the folder
can be pushed to a Space as-is:

```yaml
sdk: static
app_file: index.html
header: mini
```

`sdk_version` / `python_version` / `app_port` are deliberately absent — those
are Gradio and Docker fields, and there is no Python app here. Colours come
from HF's fixed palette. Validated against the
[configuration reference](https://huggingface.co/docs/hub/spaces-config-reference):
no unknown keys, and `app_file` resolves.

#### Keep the prologue ASCII

Everything before `<head>` in the page must be ASCII, and the builder fails if
it is not. Hugging Face injects a `window.huggingface = {...}` script into the
served HTML, and its insertion offset does not survive multi-byte UTF-8 in the
prologue: computed in one unit and applied in the other, so each multi-byte
character shifts it earlier. **One em dash in the licence comment** — 3 bytes,
1 character — drifted it by two and landed the script inside the opening tag:

```html
<html lang="en">
  <hea<script>window.huggingface={…};</script>d>
```

The browser parsed `<hea<script>` as a bogus tag and rendered the leftover
`d>` as text at the top of the page. Our markup was valid; the drift was
entirely in the prologue's byte count, which is why the page worked on every
other host. Diagnosed from the deployed Space:
the insertion sat exactly 2 characters early, matching the prologue's 2-byte
drift. `index.html` and `vision-bench.html` carried the same em dash and were
fixed too, since either could be deployed later.

There is also deliberately **no `custom_headers`** block. Setting COOP/COEP
would cross-origin isolate the page, which it does not need (the wasm is
single-threaded, no `SharedArrayBuffer`) and which would break the two
cross-origin resources it does use — the GitHub avatar and the mesh module from
esm.sh. Worth knowing before anyone adds them for wasm threads.

Cross-device rooms there need **HTTPS** — `RTCPeerConnection` is secure-context
only, with `localhost` the sole exception — which Spaces, Pages and Netlify all
provide. The wasm bundle is 8.6 MB, under HF's 10 MB LFS advisory.

### Serving it for another device

`serve.py` binds `0.0.0.0` by default and prints the LAN URL. One catch decides
whether the mesh can come up at all:

```sh
python3 crates/bindings/rlx-web/serve.py --tls --open cluster.html
#   bind:    0.0.0.0:8000
#   LAN:     https://10.0.0.185:8000/cluster.html
```

**`RTCPeerConnection` requires a secure context.** `localhost` is exempt, so
tabs on this machine work over plain http — but a phone loading
`http://<lan-ip>/cluster.html` gets no WebRTC at all, and the page falls back to
same-browser tabs, which defeats the point of binding `0.0.0.0`. `--tls` serves
https with a self-signed cert (generated once via `openssl`, with the LAN IP as
a SAN so the address matches); the browser warns once. The deployed static site
is https anyway, so this only matters for local testing.

### Joining from another device

The page is a **static site** — nothing of ours runs at request time, so it
deploys to GitHub Pages as-is. Transport depends on reach:

| Reach | Transport | Setup |
|-------|-----------|-------|
| Tabs in one browser | `BroadcastChannel` | none — just open another tab |
| Other devices | **WebRTC mesh** | press *Open a room*, share the join URL |

In a room, signalling (an offer, an answer, a few ICE candidates — kilobytes,
once per peer) goes over public relays, and from then on every gradient bucket
travels **directly peer-to-peer**. That split is the reason for WebRTC rather
than piping the cluster through a public broker or an iroh relay: buckets are
tens to hundreds of KB *every step*, which is abusive to relay and slow to
route, while signalling is tiny and one-shot.

Mesh and signalling come from [Trystero](https://github.com/dmotz/trystero)
(MIT), loaded as a pinned ES module. Its 0.26 API is *not* the tuple-returning
one most examples show — `makeAction` returns an object and `onPeerJoin` is a
property to assign — and getting that wrong surfaced in a browser only as
"mesh unavailable ({} is not iterable)". `tests/cluster_mesh.test.mjs` pins the
shape against a fake module, accepting both the object and the legacy tuple, so
a version bump fails in the gate instead. WebRTC negotiation is the one part of this
demo that cannot be tested outside a browser, so it belongs in maintained code
rather than ours. `?p2p=<url>` overrides the module, so pointing at a vendored
or self-hosted copy is a URL change, not a code change; `?relays=` overrides the
signalling relays.

### Getting the room onto a second device

The page shows the join URL and a **QR** of it. The encoder
([`qrcode-generator`](https://www.npmjs.com/package/qrcode-generator), MIT, no
dependencies) is *vendored* rather than fetched from a CDN — it is core UX for
the one thing the page cannot do without, and because it runs in Node,
`tests/cluster_qr.test.mjs` checks the matrix directly: plausible module count,
a sane dark ratio, and the three finder patterns in the right corners (which
catches a transposed or offset matrix). Exactly one line is added to the
published file, noted in its header.

### Calling the model from Rust, JavaScript or Python

The page ends with worked examples in all three — inference and training — for
the same graph, and all three now **describe the model in code**.

That required a new binding. Every JS entry point in this crate used to hand
back a model defined in Rust (`mlp_forward` has its topology baked in,
`VisionBench` takes a slug), so the browser could *run* RLX but not *describe*
anything to it, while Rust and `pyrlx` both build graphs op by op.
[`graph_api.rs`](src/graph_api.rs) closes that: `WebGraph` builds nodes from
JS, `compile_forward` / `compile_backward` compile them for any browser
backend, and `feed`/`run`/`output_sizes` move data. Inputs go by name rather
than as a `run` argument because wasm-bindgen has no clean way to marshal a JS
map of `Float32Array`s.

Verified by executing the page's own snippet in Chrome against the served
bundle: 13 nodes, gradient sizes `[1, 100352, 128, 1280, 10]` exactly as its
comment claims, a forward loss of `2.302585` — `ln(10)`, which is what a fresh
10-class classifier must give — and a falling loss over 100 steps. A smaller
check confirms the arithmetic: `x·w` with every weight `0.5` returns the halved
row sums.

One structural detail worth knowing when extending this: constructing a
`JsValue` **aborts** outside wasm, so an error path returning one cannot be
tested natively at all. Backend-name parsing is therefore a pure
`Result<_, String>` function with the conversion at the boundary, which keeps
the refusal and its wording checkable.

The shape is the same in each: build a graph, let autodiff rewrite it into a
backward graph whose outputs are `[loss, grads…]`, compile for a device, feed
`d_output = 1` as the seed, apply the optimizer. Changing `Device::Cpu` to
`Metal`, `Cuda` or `WebGpu` changes nothing else.

### Diagnosing "could not connect to peer"

That message has three causes needing three different answers, and they are
indistinguishable from the text alone — so the page shows the ICE state per
peer: connection state, the selected candidate pair, and **what was gathered**.

| What you see | Cause | Fix |
|---|---|---|
| `gathered host×N` only | no STUN — peer knows only its LAN addresses | fixed: STUN is now default |
| preflight `srflx ✗` | STUN itself unreachable (UDP blocked) | nothing here can fix it; the peer is unreachable off-network |
| peer never reaches `connected`, fresh id each attempt | the peer is gone (a tab closed without a clean leave stays advertised on a public relay) or suspended (a backgrounded tab is throttled too hard to finish ICE) | nothing to fix; it is abandoned after `handshakeTimeoutMs` |
| `srflx` gathered on both sides, still no pair | symmetric NAT both ends — no direct path exists | `?turn=…`, which relays every byte |

An **ICE preflight** runs before any peer exists — a throwaway
`RTCPeerConnection` that gathers candidates and reports `host`/`srflx`/`relay`
with the reflexive address. That is what separates the three cases without
guessing, and it needs no peer and no room. It also makes one thing visible
that otherwise looks like NAT: two peers reporting the *same* reflexive address
are behind the same NAT and should be pairing on host candidates. (That line
contains your public IP — worth knowing before sharing a screenshot.)

Per peer, the page also shows how many **remote candidates** arrived. Zero
after a completed SDP exchange is the signature of a peer that is no longer
there, as opposed to one that cannot be routed to.

The failure is **per peer**, not per room: `JoinError` carries a `peerId`, and
the cohort keeps working with whoever did connect. Reporting it as "join
failed" sent the reader hunting for a NAT problem when the usual cause is a
peer that no longer exists, so the message now names the peer, states how many
peers *are* connected, and only raises TURN when nothing has connected at all.
Repeat failures for the same peer are reported once.

### ICE: STUN by default, TURN never

Without STUN a peer gathers only `host` candidates — its own LAN addresses.
Two tabs on one machine and two devices on one network both work that way,
which is why the gap only appears across networks: the SDP exchange succeeds
and ICE then fails with *"could not connect to peer … configure TURN servers"*.
Signalling looks perfectly healthy.

Public STUN is now configured by default (`DEFAULT_ICE_SERVERS`), and verified
in a browser by reading `getStats()` — a rank gathers `srflx` candidates where
it previously had only `host`. STUN only tells a peer its own public address;
**nothing is relayed through it**, so "gradients never touch a third party"
still holds.

#### There is no usable public TURN (tested 2026-10-12)

Every candidate was tested, not taken from a list — DNS, ports, and then the
page's own ICE preflight checking for actual `relay` candidates:

| Endpoint | Result |
|----------|--------|
| `openrelay.metered.ca` (Open Relay Project — what every list recommends) | resolves to `192.0.2.1`, an RFC 5737 *documentation* address. Parked; 80/443/3478 all closed |
| `relay.metered.ca` (its successor) | alive; rejects the legacy `openrelayproject` credentials → account needed |
| `freeturn.net` | no DNS |
| `turn.anyfirewall.com`, `numb.viagenie.ca` | no DNS |
| `speed.cloudflare.com/turn-creds` | **origin-gated**: 403 unless `Origin` is `https://speed.cloudflare.com`, which a browser can never forge cross-origin. Not usable from any other site |
| `rstream.io` | credential endpoints are per-project → account needed |
| `turn.cloudflare.com` | alive; short-lived credentials via an authenticated API. Free tier is 1,000 GB/month egress shared with Realtime SFU |

Public **STUN** is still plentiful and is configured by default, verified by the
preflight gathering `srflx` candidates. The asymmetry is the whole story: STUN
reports an address and relays nothing, while TURN carries every byte. That is
what killed the free ones, and it is why the next list promising otherwise will
also be stale.

#### A static site can still use TURN

Being static rules out *minting* credentials at runtime — that needs a secret,
which must never ship in a page. It does **not** rule out using them: mint once
out of band and put the result in the share URL, which the QR already carries.

```sh
export TURN_KEY_ID=… TURN_KEY_API_TOKEN=…        # Cloudflare dashboard → Realtime → TURN
python3 crates/bindings/rlx-web/tools/turn.py --cloudflare --ttl 3600
# → &turn=turn:turn.cloudflare.com:3478?transport=udp|<user>|<cred>;turns:…
```

Nothing of ours runs at request time; the credential expires on its own. Note
the link is a bearer secret until it does.

The cheapest fix remains no relay at all: on **one network** host candidates
pair directly, which is the 4-rank run across Chrome, Firefox and Safari
recorded above. [`tools/turn.py`](tools/turn.py) with no flags runs a local
coturn for the case where the devices can reach each other but NAT will not let
them pair — though not for Wi-Fi client isolation, where the relay is just
another blocked client.

TURN is a different bargain and therefore never a default. When both peers sit
behind symmetric NAT there is no direct path and every byte goes through the
relay — here that is the whole parameter vector each round, tens of MB/s. Supply
your own if you need it:

```text
?turn=turn:host:3478|user|credential;turns:host:5349|user|credential
?ice=stun:host:3478,stun:other:3478      # replaces the STUN defaults
```

TURN entries are appended *after* STUN so a direct path is always tried first.
When ICE fails, the UI now says which case it is rather than just "mesh
unavailable".

`RTCPeerConnection` does not exist in a Web Worker, so the **main thread owns
the transport** while the rank and the protocol stay in the Worker. The Worker
talks to a `proxyBus` that encodes each message to one ArrayBuffer and
*transfers* it — the UI thread moves buckets without copying or parsing them,
and compute never leaves the Worker.

Both transports share one wire format
([`web/cluster-bus.js`](web/cluster-bus.js)):

```text
[u32 LE header length][JSON header, space-padded to a multiple of 4][f32 payload]
```

The padding is load-bearing. `new Float32Array(buffer, offset)` throws unless
`offset` is a multiple of 4, and a JSON header is whatever length it happens to
be — so the header is padded with spaces (still valid JSON) rather than copying
a few hundred KB into a fresh aligned buffer every step.

Bandwidth is what limits a mesh: the leader sends parameters to **every** peer
each step, so the page defaults to `mnist-cnn` (5,258 params ≈ 21 KB a step)
rather than `mnist-mlp` (101,770 params ≈ 398 KB, over a data channel's message
limit and therefore chunked). See *Scaling* below.

### Verification

| What | How |
|------|-----|
| Sharded gradients average to the full-batch gradient | `cargo test -p rlx-web --lib cluster` — two half-batch buckets vs one full batch, `< 1e-5`. True only because every loss reduces over the batch with `ReduceOp::Mean`; a `Sum` would silently scale the effective learning rate with the tab count |
| Ranks stay bit-identical; one leader elected | `node --test crates/bindings/rlx-web/tests/cluster_protocol.test.mjs` |
| Elastic join, leader takeover, split-brain convergence, stopped-tab exclusion | same — against a stub rank over an in-process bus with structured-clone semantics |
| A mixed cohort tolerates a rank that answers asynchronously (a WebGPU rank among CPU ranks) | same — a stub whose `grads` resolves on a later task |
| The wire format at every header alignment; a non-finite loss crossing JSON; a cohort and an elastic join through the encoded proxy hop with real transfer semantics | `node --test crates/bindings/rlx-web/tests/cluster_bus.test.mjs` |

Both are in `just check-wasm` (the Node tests are skipped where node is absent).

**Not yet validated in a browser.** The protocol, the wire format and the proxy
hop are tested in Node, and the bundle serves with the right MIME types — but no
run in a real browser has been recorded here yet, so WebGPU ranks are compile-
and protocol-verified rather than measured. WebRTC negotiation cannot run
outside a browser at all, so the mesh is the least-verified layer here: the
tests cover everything on either side of it.

### Scaling

The design is a **star inside a full mesh**: one leader averages and
broadcasts, and WebRTC connects every peer to every other. That is fine for a
handful of ranks and will not reach a hundred:

| N | Mesh connections | Leader egress per step (`mnist-cnn`) |
|---|------------------|--------------------------------------|
| 2 | 1 | 21 KB |
| 10 | 45 | 189 KB |
| 100 | 4,950 | 2.1 MB |

Two of those costs now adapt to the live peer count
([`policyFor`](web/cluster-protocol.js)):

- **Quorum, not unanimity.** Up to three ranks the step waits for every bucket;
  beyond that it resolves once ~60% have arrived. Unanimity means one straggler
  in a hundred costs the full collect window *every step* — a loss curve that
  looks healthy while advancing at a crawl. The reduce was already
  sample-weighted over whoever arrived, and every included bucket is checked
  against the step's parameter version, so this costs nothing in correctness.
- **Heartbeat period.** Mesh-wide beat traffic grows as n²/period, so the period
  stretches with the cohort; `drop` and `takeover` are derived from the
  stretched beat, or a larger cohort would evict members that are merely quiet.

Both are reported in the UI rather than left as hidden heuristics, and guarded
by `the policy adapts to the number of peers` and `one slow rank does not
throttle a larger cohort`.

What does *not* adapt is the topology, and that is the real ceiling: at 10
steps/s a 100-rank cohort still needs ~21 MB/s upstream from one browser tab and
99 `RTCPeerConnection`s per peer. Getting there means an aggregation tree (each
node averages its subtree, so leader egress becomes O(1) instead of O(n)), a
partial mesh instead of a full one, and most likely trading lockstep for
periodic averaging.

Reaching machines that are not on one LAN is the step after that: `IrohTransport`
(`crates/core/rlx-driver/src/iroh_transport.rs`) is a drop-in `Transport` and
iroh has browser support, so a tab could join a cohort that includes native
CUDA and Metal ranks — but it is relay-only in the browser, and the `Transport`
trait is synchronous (`send_bytes`/`recv_bytes`) over a tokio multi-thread
runtime, neither of which exists on wasm. That needs an async transport seam.

### TypeScript / JavaScript SDK

The [`web/rlx.js`](web/rlx.js) wrapper (types in [`web/rlx.d.ts`](web/rlx.d.ts),
source in [`web/rlx.ts`](web/rlx.ts)) provides a developer-friendly API over the
raw wasm-bindgen exports:

```js
import Rlx from "./rlx.js";

const rlx = await Rlx.init({ webgpu: true });
const model = rlx.vision("mnist-cnn");
const params = model.initParams(42);
const { x, label } = model.syntheticBatch(1);
const logits = model.forwardCpu(x, params);
const { loss, params: next } = model.trainStepCpu(x, label, params, 0.01);
const bench = model.benchCpu(50, 42, 0.01);
```

Low-level wasm exports remain available as `rlx.wasm` / `./pkg/rlx_web.js`.
`just build-web` emits `./pkg/rlx_web.d.ts` via wasm-bindgen `--typescript`.

## JS API (low-level)

```js
import init, * as rlx from "./pkg/rlx_web.js";
await init();                                  // loads wasm; installs panic hook

// CPU (sync)
rlx.mlp_forward(x, in, hid, out, w1, b1, w2, b2);             // -> y
rlx.mlp_grads(x, t, in, hid, out, w1, b1, w2, b2);           // -> [loss, ∂w1…, ∂b1…, ∂w2…, ∂b2…]
rlx.mlp_train_step(x, t, in, hid, out, w1, b1, w2, b2, lr);  // -> updated params

// WebGPU (async, --webgpu)
await rlx.init_webgpu();                                      // -> bool
await rlx.mlp_forward_gpu(x, in, hid, out, w1, b1, w2, b2);
await rlx.mlp_grads_gpu(x, t, in, hid, out, w1, b1, w2, b2);

// WebGL2 (sync, --webgl)
rlx.mlp_forward_webgl(x, in, hid, out, w1, b1, w2, b2);
rlx.mlp_grads_webgl(x, t, in, hid, out, w1, b1, w2, b2);
```

## CI gate

`just check-wasm` compiles the CPU + WebGPU + WebGL stack for
`wasm32-unknown-unknown` (all feature configs) and runs `rlx-webgl`'s native
parity tests. It is part of `just ci`.

## License

MIT OR Apache-2.0.
