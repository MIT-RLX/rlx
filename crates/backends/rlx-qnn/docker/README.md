# rlx-qnn — Docker validation

Closes the rlx-qnn loop in a Linux container: build the emitted `qnn_model.cpp`
with `qnn-model-lib-generator` and run it on the QNN **x86 reference backend**
(`libQnnCpu.so`) — no Snapdragon device needed.

The Qualcomm AI Engine Direct (QNN) SDK is **proprietary** and gated behind a
Qualcomm account (Qualcomm Package Manager). There is no public image to pull,
so the SDK is **never baked into an image** — you download it once and it gets
mounted read-only into the container at run time.

## Files

| File | What it is |
| --- | --- |
| `validate.py` | Cross-platform driver (macOS/Linux/Windows). Two modes below. |
| `Dockerfile` | Ubuntu + clang + numpy; `ENTRYPOINT` runs `run_qnn.sh`. SDK mounted at `/opt/qnn`. |
| `mock_net_run.py` | numpy stand-in for `qnn-net-run`, used only by the SDK-free self-test. |

## `harness-test` — SDK-free (needs only Docker)

Validates the **host harness** (`verify.py` data-gen → `input_list` → raw
layout → reshape → parity check) end-to-end, with a numpy stand-in for
`qnn-net-run`. It does **not** exercise the QNN C++ lowering — but it proves
every file-format/plumbing contract around it. CI-friendly.

```sh
python3 crates/rlx-qnn/docker/validate.py harness-test --dims 8 16 4
# or, from the repo root:  just qnn-docker-test 8 16 4
```

## `run` — real validation (needs the QNN SDK)

Builds the model lib and runs it on `libQnnCpu.so` inside the container.

```sh
export QNN_SDK_ROOT=/path/to/qairt/2.x        # the SDK you downloaded
python3 crates/rlx-qnn/docker/validate.py run --dims 32 64 32
# or:  just qnn-docker-run 32 64 32
```

Swap the backend in `run_qnn.sh` from `libQnnCpu.so` to `libQnnHtp.so` to target
the Hexagon NPU on real silicon (where the perf lives).

## Notes

- `--artifacts DIR` validates a pre-emitted artifact dir instead of running
  `rlx-qnn-emit` (handy while the workspace is mid-build).
- Both modes emit fresh artifacts via `cargo run -p rlx-qnn --bin rlx-qnn-emit`
  by default, so they always test the current codegen.
## License

MIT OR Apache-2.0.
