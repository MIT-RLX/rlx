#!/usr/bin/env bash
# Validation entry point inside the Docker container.
#
# Runs in three layers:
#   1. `cargo test -p rlx-tpu --release` — host-agnostic unit/smoke
#      + (when LIBTPU_PATH is set, which it is in this image) the
#      pjrt_roundtrip + pjrt_bench tests against the real plugin.
#   2. `cargo test -p rlx-runtime --features cpu,tpu` —
#      compile-cache integration test that exercises Device::Tpu
#      through Session.
#   3. `validate_hlo.py` — round-trips emitted HLO bytes through
#      jax.lib.xla_extension.HloModule for parse validity.

set -euo pipefail

cd /work

echo "═══════════════════════════════════════════════════════"
echo "[1/3] cargo test -p rlx-tpu --release"
echo "═══════════════════════════════════════════════════════"
cargo test -p rlx-tpu --release -- --nocapture --test-threads=1

echo
echo "═══════════════════════════════════════════════════════"
echo "[2/3] cargo test -p rlx-runtime --features cpu,tpu"
echo "═══════════════════════════════════════════════════════"
cargo test -p rlx-runtime --release --features cpu,tpu \
    --test tpu_compile_cache -- --nocapture --test-threads=1

echo
echo "═══════════════════════════════════════════════════════"
echo "[3/3] HLO byte validation via jax.xla_extension"
echo "═══════════════════════════════════════════════════════"

cargo run --release --quiet \
    -p rlx-tpu --example emit_hlo_samples 2>&1 | tee /tmp/hlo_samples.list

python3 /work/rlx-tpu/docker/validate_hlo.py /tmp/hlo_samples.list

echo
echo "✓ all rlx-tpu validation passed"
