// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

using Rlx;

/// Inference: build a graph, compile it for a device, set its weights, run it.
/// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
static class Inference
{
    static float[] Weights(int n, int seed) =>
        Enumerable.Range(0, n).Select(i => ((i * 7919 + seed * 104729) % 1000) / 1000f - 0.5f).ToArray();

    public static void Run(string[] args)
    {
        var device = args.Length > 0 ? args[0] : "cpu";
        var g = new RLX.Graph("classifier");
        var x = g.Input("x", new long[] { 1, 4 });
        var hidden = g.Relu(g.Matmul(x, g.Param("w1", new long[] { 4, 8 })));
        g.SetOutputs(new[] { g.Softmax(g.Matmul(hidden, g.Param("w2", new long[] { 8, 3 }))) });

        using var model = new RLX.Session(device).Compile(g); // consumes g
        Console.WriteLine($"compiled on: {model.Device()}");
        model.SetParams(new Dictionary<string, float[]> { ["w1"] = Weights(32, 1), ["w2"] = Weights(24, 2) });
        var probs = model.Run(new Dictionary<string, float[]> { ["x"] = new[] { 0.5f, -1, 2, 0.25f } })[0];
        Console.WriteLine("probs: " + string.Join(" ", probs.Select(p => p.ToString("F4"))));
        Console.WriteLine($"probs sum: {probs.Sum():F4}");
        Console.WriteLine($"output shapes: [{string.Join(", ", model.OutputShapes().Select(s => "[" + string.Join(", ", s) + "]"))}] dtypes: [{string.Join(", ", model.OutputDtypes())}]");
    }
}
