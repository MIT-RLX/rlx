// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//   ldc2 -i -Isource examples/backends.d -L-lrlx_capi && ./backends
module backends;

import std.algorithm : max;
import std.array : join;
import std.math : abs;
import std.stdio : writefln, writeln;
import rlx;

Graph model()
{
    auto g = new Graph(name: "mlp");
    immutable x = g.input("x", [4, 16]);
    g.setOutputs([g.tanh(g.matmul(x, g.param("w", [16, 16])))]);
    return g;
}

void main()
{
    float[] w, xs;
    foreach (i; 0 .. 256)
        w ~= (i % 5) / 10.0f - 0.2f;
    foreach (i; 0 .. 64)
        xs ~= (i % 7) / 7.0f;
    const(float[])[string] x = ["x": xs];

    auto available = devices();
    writeln("devices: ", available.join(" "));
    writeln("metal available: ", isAvailable("metal"));
    writeln("fastest for graph: ", fastestDeviceFor(model()));
    float[] reference;
    foreach (device; available)
    {
        auto c = new Session(device: device).compile(model());
        c.setParam("w", w);
        auto y = c.run(x)[0];
        if (reference is null)
            reference = y;
        float d = 0;
        foreach (i; 0 .. y.length)
            d = max(d, abs(y[i] - reference[i]));
        writefln("session %s: max|Δ| vs first = %.2e", device, d);
    }
    auto flex = new FlexibleSession(policy: parseJSON(`{"prefer": ["metal", "cpu"]}`));
    writeln("flexible session picked: ", flex.compile(model()).device());
    auto runner = new Runner(model());
    runner.setParam("w", w);
    writeln("runner warmed: ", runner.warmAll().join(" "));
    foreach (row; runner.benchmark(x, runs: 3).array)
        writeln("  bench ", row["device"].str, ": exec_ms=", row["exec_ms"]);
    writeln("runner ran on: ", runner.run(x).device);
    writeln("run_try chain: ", runner.runTry(["metal", "cpu"], x).device);
    auto router = new Router(model());
    router.setParam("w", w);
    writeln("router ran on: ", router.runChain(x).device);
}
