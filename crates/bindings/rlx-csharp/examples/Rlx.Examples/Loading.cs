// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

using System.Text;
using Rlx;

/// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
/// into a compiled graph, and quantizing in memory. RLX_TOKENIZER=&lt;tokenizer.json&gt;
/// adds the tokenizer.
static class Loading
{
    public static void Run()
    {
        var w = Enumerable.Range(0, 64).Select(i => (i % 9) / 4f - 1).ToArray();
        var gguf = Path.Combine(Path.GetTempPath(), "rlx-example-cs.gguf");
        using (var writer = new RLX.GgufWriter())
        {
            writer.AddF32("proj.weight", new long[] { 8, 8 }, "F32", w);
            writer.AddF32("proj.weight_q8", new long[] { 8, 8 }, "Q8_0", w);
            writer.Write(gguf, architecture: "demo", metadata: new Dictionary<string, object> { ["demo.layers"] = 1 });
        }
        float[] exact;
        using (var f = RLX.Gguf.Open(gguf))
        {
            Console.WriteLine($"gguf architecture: {f.Architecture()}");
            Console.WriteLine($"gguf tensors: {string.Join(" ", f.TensorNames())}");
            exact = f.Tensor("proj.weight");
            var q8 = f.Tensor("proj.weight_q8");
            Console.WriteLine($"gguf f32 exact: {exact.SequenceEqual(w)} q8 max|Δ|: {exact.Zip(q8, (a, b) => Math.Abs(a - b)).Max():F4}");
        }

        var g = new RLX.Graph("proj");
        var x = g.Input("x", new long[] { 1, 8 });
        g.SetOutputs(new[] { g.Matmul(x, g.Param("proj.weight", new long[] { 8, 8 })) });
        var c = new RLX.Session().Compile(g);
        c.SetParam("proj.weight", exact);
        var y = c.Run(new Dictionary<string, float[]> { ["x"] = Enumerable.Repeat(1f, 8).ToArray() })[0];
        Console.WriteLine($"gguf -> graph: {y[0]:F2} {y[1]:F2} {y[2]:F2}");

        var st = Path.Combine(Path.GetTempPath(), "rlx-example-cs.safetensors");
        var header = $"{{\"proj.weight\":{{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,{4 * w.Length}]}}}}";
        header += new string(' ', (8 - header.Length % 8) % 8);
        using (var fs = File.Create(st))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((ulong)header.Length);
            bw.Write(Encoding.UTF8.GetBytes(header));
            foreach (var v in w) bw.Write(v);
        }
        var loaded = RLX.LoadWeights("safetensors", st);
        Console.WriteLine($"safetensors: [{string.Join(", ", loaded.Keys)}] exact: {loaded["proj.weight"].SequenceEqual(w)}");

        var packed = RLX.Quantize(w, "Q8_0");
        Console.WriteLine($"quantize q8_0: {4 * w.Length} -> {packed.Length} bytes, round trip {RLX.Dequantize(packed, "Q8_0", w.Length).Length} values");
        File.Delete(gguf);
        File.Delete(st);

        var tokPath = Environment.GetEnvironmentVariable("RLX_TOKENIZER");
        if (tokPath is not null)
        {
            using var tok = RLX.Tokenizer.Load(tokPath);
            var ids = tok.Encode("Hello from C#");
            Console.WriteLine($"tokenizer: {ids.Length} ids -> \"{tok.Decode(ids)}\"");
        }
        else Console.WriteLine("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)");
        Console.WriteLine($"argmax: {RLX.Argmax(new[] { 0.1f, 2, 0.3f })}");
    }
}
