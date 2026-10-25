// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

using Rlx;

/// Picking a backend: what this build has, what suits a graph, and the ways to
/// choose — by name, by cost model, by policy, by benchmark, per run.
static class Backends
{
    static RLX.Graph Model()
    {
        var g = new RLX.Graph("mlp");
        var x = g.Input("x", new long[] { 4, 16 });
        g.SetOutputs(new[] { g.Tanh(g.Matmul(x, g.Param("w", new long[] { 16, 16 }))) });
        return g;
    }

    public static void Run()
    {
        var x = new Dictionary<string, float[]> { ["x"] = Enumerable.Range(0, 64).Select(i => (i % 7) / 7f).ToArray() };
        var w = Enumerable.Range(0, 256).Select(i => (i % 5) / 10f - 0.2f).ToArray();
        var available = RLX.Devices();
        Console.WriteLine($"devices: {string.Join(" ", available)}");
        Console.WriteLine($"metal available: {RLX.IsAvailable("metal")}");
        Console.WriteLine($"fastest for graph: {RLX.FastestDeviceFor(Model())}");
        float[]? reference = null;
        foreach (var device in available)
        {
            var c = new RLX.Session(device).Compile(Model());
            c.SetParam("w", w);
            var y = c.Run(x)[0];
            reference ??= y;
            Console.WriteLine($"session {device}: max|Δ| vs first = {y.Zip(reference, (a, b) => Math.Abs(a - b)).Max():0.00e+00}");
        }
        var flex = new RLX.FlexibleSession(new { prefer = new[] { "metal", "cpu" } });
        Console.WriteLine($"flexible session picked: {flex.Compile(Model()).Device()}");
        var runner = new RLX.Runner(Model());
        runner.SetParam("w", w);
        Console.WriteLine($"runner warmed: {string.Join(" ", runner.WarmAll())}");
        foreach (var row in runner.Benchmark(x, 3)) Console.WriteLine($"  bench {row!["device"]}: exec_ms={row["exec_ms"]}");
        Console.WriteLine($"runner ran on: {runner.Run(x).Device}");
        Console.WriteLine($"run_try chain: {runner.RunTry(new[] { "metal", "cpu" }, x).Device}");
        var router = new RLX.Router(Model());
        router.SetParam("w", w);
        Console.WriteLine($"router ran on: {router.RunChain(x).Device}");
    }
}
