# Off-TPU validation harness for rlx-tpu.
#
# Validates two things, neither of which need TPU hardware:
#
#   1. **Linux build cleanliness** — `cargo test -p rlx-tpu` runs the
#      host-agnostic suite (proto encoder, HLO builder, lowering walker,
#      smoke graphs). The PJRT round-trip tests skip cleanly without
#      LIBTPU_PATH.
#
#   2. **HLO structural validity** — `validate_hlo.py` deserializes the
#      `HloModuleProto` bytes our lowering emits via JAX's `xla_extension`
#      module. If the bytes parse without error and the resulting
#      `HloModule` has the expected shape, our proto field numbers /
#      dimension order / opcodes are correct.
#
# What we deliberately do NOT validate here:
#
#   * Numerical execution. JAX / jaxlib does not ship a standalone
#     `libpjrt_c_cpu.so` (XLA's CPU PJRT plugin is statically linked
#     into `xla_extension.so`, not exposed as a dlopen target). For
#     end-to-end execution validation, point `LIBTPU_PATH` at a real
#     libtpu plugin on a GCP TPU VM — the same Rust / HLO path runs.
#
# To build + run:
#   docker build -t rlx-tpu-validate -f rlx-tpu/docker/Dockerfile .
#   docker run --rm -v $(pwd):/work -w /work rlx-tpu-validate \
#       /work/rlx-tpu/docker/run-tests.sh

# Edition 2024 + `if let` chains (stabilized in Rust 1.88) are both
# used by the workspace.
FROM rust:1.89-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# JAX gives us the `xla_extension` module which can parse HLO bytes
# back into an HloModule object. We only use the parsing path —
# validates proto field numbers + dimension layout + opcode strings
# without needing a runnable PJRT plugin.
RUN python3 -m venv /opt/jax-venv && \
    /opt/jax-venv/bin/pip install --no-cache-dir \
        "jax==0.4.34" "jaxlib==0.4.34"

ENV PATH="/opt/jax-venv/bin:${PATH}"

WORKDIR /work
CMD ["/work/rlx-tpu/docker/run-tests.sh"]
