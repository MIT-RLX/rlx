#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Inline an ONNX model's EXTERNAL-DATA weights into a single self-contained
# .onnx so rlx-onnx-import (whose `onnx` 0.1.0 proto lacks the external_data /
# data_location fields) can read them from `raw_data`.
#
# Many large exports (LM+codec TTS: moss-nano, chatterbox, …) ship a tiny .onnx
# graph with weights in sidecar `.data` / `.onnx_data` files referenced by each
# initializer's `external_data` field. rlx reads only inline `raw_data`, so those
# weights arrive empty. This loads the external data and rewrites it inline.
#
# ⚠️ protobuf caps a single message at 2 GiB — inlining a >2 GiB model fails; those
# need the rust-side external-data reader (offset/length sidecar loading) instead.
#
# Usage: python inline_external_data.py in.onnx out.onnx
import sys

import onnx


def main(src: str, dst: str) -> None:
    m = onnx.load(src, load_external_data=True)  # pulls sidecar bytes into raw_data
    # Force every initializer back to inline storage (clear any external refs).
    for init in m.graph.initializer:
        if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
            init.ClearField("external_data")
            init.data_location = onnx.TensorProto.DEFAULT
    onnx.save(m, dst, save_as_external_data=False)
    print(f"inlined {dst}: {len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers (self-contained)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: inline_external_data.py <in.onnx> <out.onnx>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
