# rlx-qnn validation image — build + run the emitted QNN model on the x86
# reference backend with the Qualcomm AI Engine Direct (QNN) SDK.
#
# The QNN SDK is proprietary and NOT redistributable, so it is NOT baked into
# this image. Mount it at runtime instead:
#
#   docker run --rm \
#     -v /path/to/qairt/2.x:/opt/qnn:ro \   # the SDK (QNN_SDK_ROOT)
#     -v /path/to/qnn-out:/work \           # rlx-qnn-emit artifacts
#     rlx-qnn-validate
#
# `crates/rlx-qnn/docker/validate.py` drives the whole thing for you.
FROM ubuntu:22.04

# Toolchain the SDK's qnn-model-lib-generator / qnn-net-run need on x86 Linux,
# plus numpy for the verify.py host harness.
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang \
        libc++-dev \
        libc++abi-dev \
        make \
        python3 \
        python3-numpy \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Mount the SDK here at runtime (-v /path/to/qnn:/opt/qnn:ro).
ENV QNN_SDK_ROOT=/opt/qnn

# Mount the rlx-qnn-emit artifact dir here (-v /path/to/qnn-out:/work).
WORKDIR /work

# run_qnn.sh sources $QNN_SDK_ROOT/bin/envsetup.sh, builds the model lib, runs
# it on libQnnCpu.so, and checks the output against numpy.
ENTRYPOINT ["bash", "run_qnn.sh"]
