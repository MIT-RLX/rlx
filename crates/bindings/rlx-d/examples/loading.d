// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.
//   ldc2 -i -Isource examples/loading.d -L-lrlx_capi && ./loading
module loading;

import std.algorithm : map, max;
import std.array : join, replicate;
import std.bitmanip : nativeToLittleEndian;
import std.file : remove, tempDir, write;
import std.format : format;
import std.math : abs;
import std.path : buildPath;
import std.stdio : writefln, writeln;
import rlx;

void main()
{
    float[] w;
    foreach (i; 0 .. 64)
        w ~= (i % 9) / 4.0f - 1;

    immutable gguf = buildPath(tempDir, "rlx-example-d.gguf");
    auto writer = new GgufWriter();
    writer.addF32("proj.weight", [8, 8], "F32", w);
    writer.addF32("proj.weight_q8", [8, 8], "Q8_0", w);
    writer.write(gguf, architecture: "demo", metadata: parseJSON(`{"demo.layers": 1}`));
    auto f = Gguf.open(gguf);
    writeln("gguf architecture: ", f.architecture());
    writeln("gguf tensors: ", f.tensorNames().join(" "));
    auto exact = f.tensor("proj.weight");
    auto q8 = f.tensor("proj.weight_q8");
    float maxd = 0;
    foreach (i; 0 .. 64)
        maxd = max(maxd, abs(exact[i] - q8[i]));
    writefln("gguf f32 exact: %s q8 max|Δ|: %.4f", exact == w, maxd);

    auto g = new Graph(name: "proj");
    immutable x = g.input("x", [1, 8]);
    g.setOutputs([g.matmul(x, g.param("proj.weight", [8, 8]))]);
    auto c = new Session().compile(g);
    c.setParam("proj.weight", exact);
    auto y = c.run(["x": [1.0f, 1, 1, 1, 1, 1, 1, 1]])[0];
    writeln("gguf -> graph: ", y[0 .. 3].map!(v => format("%.2f", v)).join(" "));

    immutable st = buildPath(tempDir, "rlx-example-d.safetensors");
    auto header = `{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}`;
    header ~= " ".replicate((8 - header.length % 8) % 8);
    ubyte[] bytes = nativeToLittleEndian(cast(ulong) header.length).dup ~ cast(ubyte[]) header.dup;
    foreach (v; w)
        bytes ~= nativeToLittleEndian(v);
    write(st, bytes);
    auto loaded = loadWeights("safetensors", st);
    writeln("safetensors: [proj.weight] exact: ", loaded["proj.weight"] == w);

    auto packed = quantize(w, "Q8_0");
    writefln("quantize q8_0: %d -> %d bytes, round trip %d values", w.length * 4, packed.length,
        dequantize(packed, "Q8_0", count: w.length).length);
    remove(gguf);
    remove(st);
    writeln("argmax: ", argmax([0.1f, 2, 0.3]));
}
