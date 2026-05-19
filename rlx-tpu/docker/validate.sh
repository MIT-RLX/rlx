#!/usr/bin/env bash
# rlx-tpu/docker/validate.sh — drive the off-TPU validation harness.
#
# Two layers of validation, picked by flag:
#
#   ./validate.sh                # default: parse-only — fast (~1 min)
#                                #   uses jaxlib's `xla_extension` to
#                                #   deserialize the HLO bytes we emit
#                                #   and assert structural properties.
#   ./validate.sh --numerical    # full: compile + execute through XLA's
#                                #   CPU PJRT plugin, compare numbers
#                                #   to in-test references. First-time
#                                #   setup builds the plugin from
#                                #   source via Bazel — 30–90 min.
#                                #   Subsequent runs reuse the cached
#                                #   `rlx-xla-cpu-plugin` image and
#                                #   take ~1 min.
#
# Other flags:
#   --build         rebuild the parse-only image
#   --build-plugin  force-rebuild the XLA CPU plugin base image
#                   (only relevant with --numerical)
#   -- <cmd>        replace the default test command, e.g.
#                   `./validate.sh --numerical -- bash`
#
# Designed to work the same on macOS / Linux / Windows-with-Docker.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKERFILE="$REPO_ROOT/rlx-tpu/docker/Dockerfile"
DOCKERFILE_XLA="$REPO_ROOT/rlx-tpu/docker/Dockerfile.xla-cpu"
DOCKERFILE_NUM="$REPO_ROOT/rlx-tpu/docker/Dockerfile.numerical"

PARSE_IMAGE=rlx-tpu-validate
PLUGIN_IMAGE=rlx-xla-cpu-plugin
NUM_IMAGE=rlx-tpu-numerical

REBUILD=0
REBUILD_PLUGIN=0
NUMERICAL=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)        REBUILD=1; shift ;;
        --build-plugin) REBUILD_PLUGIN=1; shift ;;
        --numerical)    NUMERICAL=1; shift ;;
        --)             shift; EXTRA_ARGS+=("$@"); break ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

build_if_missing() {
    local img="$1" dockerfile="$2" force="$3"
    if [[ "$force" -eq 1 ]] || ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "[validate.sh] building image $img"
        docker build -t "$img" -f "$dockerfile" "$REPO_ROOT"
    fi
}

if [[ "$NUMERICAL" -eq 1 ]]; then
    # Build the heavy XLA CPU plugin base (one-time, cached) then the
    # thin numerical harness layered on top.
    build_if_missing "$PLUGIN_IMAGE" "$DOCKERFILE_XLA" "$REBUILD_PLUGIN"
    build_if_missing "$NUM_IMAGE"    "$DOCKERFILE_NUM" "$REBUILD"
    IMAGE="$NUM_IMAGE"
    # Three-step numerical validation:
    #   1. rlx-tpu test suite (op lowering + bench + 17 PJRT roundtrip)
    #   2. rlx-runtime compile-cache integration (TPU through Session)
    #   3. End-to-end real-model parity: same transformer block
    #      compiled on Device::Cpu and Device::Tpu, outputs compared
    #      under both AlwaysF32 and the TPU-default AutoMixedBf16
    #      policies. Catches IR-level dtype/shape regressions that
    #      single-op tests miss.
    # The real-model test (tpu_real_minilm) downloads ~22 MB from HF
    # Hub on first run and is opt-in via RLX_REAL_MODEL=1 (passed
    # through to the container below). When opted in, the host's HF
    # cache is mounted into the container so repeat runs hit the
    # cache and don't re-download.
    REAL_MODEL_TEST=""
    if [[ -n "${RLX_REAL_MODEL:-}" ]]; then
        REAL_MODEL_TEST="cargo test -p rlx-models --release \
            --features tpu,hf-download \
            --test tpu_real_minilm -- --test-threads=1 --nocapture"
    fi
    DEFAULT_CMD=(bash -c "
        set -e
        cargo test -p rlx-tpu --release -- --test-threads=1 --nocapture
        cargo test -p rlx-runtime --release --features cpu,tpu \
            --test tpu_compile_cache -- --test-threads=1 --nocapture
        cargo test -p rlx-runtime --release --features cpu,tpu \
            --test tpu_real_model_parity -- --test-threads=1 --nocapture
        cargo test -p rlx-runtime --release --features cpu,tpu \
            --test tpu_cpu_speed -- --test-threads=1 --nocapture
        cargo test -p rlx-runtime --release --features cpu,tpu \
            --test cpu_perf_diag -- --test-threads=1 --nocapture
        $REAL_MODEL_TEST
    ")
else
    build_if_missing "$PARSE_IMAGE"  "$DOCKERFILE"     "$REBUILD"
    IMAGE="$PARSE_IMAGE"
    DEFAULT_CMD=(/work/rlx-tpu/docker/run-tests.sh)
fi

if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
    CMD=("${DEFAULT_CMD[@]}")
else
    CMD=("${EXTRA_ARGS[@]}")
fi

echo "[validate.sh] running: ${CMD[*]} (image=$IMAGE)"

# Pass through bench / real-model knobs so the opt-in tests run
# inside the container when the user sets them on the host.
ENV_FLAGS=()
if [[ -n "${RLX_TPU_BENCH:-}" ]]; then
    ENV_FLAGS+=(-e "RLX_TPU_BENCH=${RLX_TPU_BENCH}")
fi
if [[ -n "${RLX_TPU_BENCH_SWEEP:-}" ]]; then
    ENV_FLAGS+=(-e "RLX_TPU_BENCH_SWEEP=${RLX_TPU_BENCH_SWEEP}")
fi
# Real-model E2E test (tpu_real_minilm) needs HF Hub access. Mount
# the host's HF cache so weights persist across runs and the
# initial download isn't repeated. Default cache dir matches what
# `hf-hub` and `transformers` use (~/.cache/huggingface). Override
# with HF_HOME on the host.
HF_VOLUME_FLAGS=()
if [[ -n "${RLX_REAL_MODEL:-}" ]]; then
    ENV_FLAGS+=(-e "RLX_REAL_MODEL=${RLX_REAL_MODEL}")
    HF_HOST_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
    mkdir -p "$HF_HOST_CACHE"
    HF_VOLUME_FLAGS+=(-v "$HF_HOST_CACHE:/root/.cache/huggingface")
fi

exec docker run --rm \
    -v "$REPO_ROOT:/work" \
    "${HF_VOLUME_FLAGS[@]}" \
    -w /work \
    "${ENV_FLAGS[@]}" \
    "$IMAGE" "${CMD[@]}"
