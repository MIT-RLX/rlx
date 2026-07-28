#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# One-shot native-vs-ort divergence localizer. Rewrites an ONNX graph so EVERY
# float intermediate tensor is also a graph OUTPUT, in topological order. Run the
# rewritten graph once through rlx (native) and once through ort, then diff
# per-tensor to find the FIRST diverging op — collapsing parity bisection from
# ~10 tap+recompile cycles to a single pair of runs.
#
# Usage:
#   python expose_all_outputs.py in.onnx exposed.onnx [names.txt]
# then run `exposed.onnx` in both ort and rlx (st_graph_native writes rlx_out_{i}
# in the SAME order as names.txt) and diff by index/name.
import sys

import onnx
from onnx import TensorProto, helper

FLOATish = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE, TensorProto.BFLOAT16}


def main(src: str, dst: str, names_path: str | None) -> None:
    m = onnx.load(src, load_external_data=True)
    # Inline any external data so the rewritten file is self-contained.
    for init in m.graph.initializer:
        if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
            init.ClearField("external_data")
            init.data_location = onnx.TensorProto.DEFAULT
    inferred = onnx.shape_inference.infer_shapes(m)
    vi = {v.name: v for v in inferred.graph.value_info}
    existing = {o.name for o in m.graph.output}
    order = []
    for n in m.graph.node:  # graph.node is topologically sorted
        for o in n.output:
            if not o or o in existing:
                continue
            v = vi.get(o)
            et = v.type.tensor_type.elem_type if v is not None else TensorProto.UNDEFINED
            if et in FLOATish:
                m.graph.output.append(helper.make_tensor_value_info(o, et, None))
                existing.add(o)
                order.append(o)
    onnx.save(m, dst, save_as_external_data=False)
    if names_path:
        with open(names_path, "w") as f:
            # existing graph outputs come first in st_graph_native's rlx_out order,
            # then the appended intermediates in topo order.
            f.write("\n".join([o.name for o in onnx.load(src, load_external_data=False).graph.output] + order))
    print(f"exposed {dst}: +{len(order)} float intermediates as outputs (total {len(m.graph.output)})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: expose_all_outputs.py <in.onnx> <exposed.onnx> [names.txt]", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
