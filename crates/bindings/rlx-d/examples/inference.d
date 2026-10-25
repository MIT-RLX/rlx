// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//   ldc2 -i -Isource examples/inference.d -L-lrlx_capi && ./inference [device]
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
module inference;

import std.algorithm : map, sum;
import std.array : join;
import std.format : format;
import std.stdio : writefln, writeln;
import rlx;

float[] weights(size_t n, size_t seed)
{
    auto w = new float[n];
    foreach (i; 0 .. n)
        w[i] = ((i * 7919 + seed * 104729) % 1000) / 1000.0f - 0.5f;
    return w;
}

void main(string[] args)
{
    immutable device = args.length > 1 ? args[1] : "cpu";
    auto g = new Graph(name: "classifier");
    immutable x = g.input("x", [1, 4]);
    immutable hidden = g.relu(g.matmul(x, g.param("w1", [4, 8])));
    g.setOutputs([g.softmax(g.matmul(hidden, g.param("w2", [8, 3])))]);

    auto model = new Session(device: device).compile(g); // consumes g
    writeln("compiled on: ", model.device());
    model.setParams(["w1": weights(32, 1), "w2": weights(24, 2)]);
    auto probs = model.run(["x": [0.5f, -1, 2, 0.25]])[0];
    writeln("probs: ", probs.map!(p => format("%.4f", p)).join(" "));
    writefln("probs sum: %.4f", probs.sum);
    writeln("output shapes: ", model.outputShapes(), " dtypes: ", model.outputDtypes());
}
