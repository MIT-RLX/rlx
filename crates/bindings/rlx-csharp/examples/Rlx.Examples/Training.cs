// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

using Rlx;

/// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
/// the same fit by hand with Grad and an Optimizer.
static class Training
{
    static readonly float[] WTrue = { 2, -1, 0.5f };
    const int Batch = 8;

    static Dictionary<string, float[]> Data(ulong step)
    {
        var x = new float[Batch * 3];
        var y = new float[Batch];
        for (int i = 0; i < Batch; i++)
            for (int k = 0; k < 3; k++)
            {
                x[i * 3 + k] = (float)(((long)step * 31 + i * 7 + k * 3) % 17) / 8.5f - 1;
                y[i] += x[i * 3 + k] * WTrue[k];
            }
        return new Dictionary<string, float[]> { ["x"] = x, ["y"] = y };
    }

    static string Fmt(float[] v) => string.Join(" ", v.Select(a => a.ToString("F3")));

    public static void Run(string[] args)
    {
        var device = args.Length > 0 ? args[0] : "cpu";
        var g = new RLX.Graph("regression");
        var x = g.Input("x", new long[] { Batch, 3 });
        var y = g.Input("y", new long[] { Batch, 1 });
        var w = g.Param("w", new long[] { 3, 1 });
        var d = g.Sub(g.Matmul(x, w), y);
        g.SetOutputs(new[] { g.Mean(g.Mul(d, d), new long[] { 0, 1 }) });

        var trainer = new RLX.Trainer(g.Clone(), new[] { "w" }, new Dictionary<string, float[]> { ["w"] = new float[3] },
            new { device, optimizer = new { kind = "adamw", lr = 0.1 } });
        var first = trainer.Evaluate(Data(0));
        trainer.Run(400, Data, lr: s => (float)(0.1 * 0.5 * (1 + Math.Cos(Math.PI * s / 400))), onStep: (_, loss) => loss > 1e-6f);
        Console.WriteLine($"trainer loss: {first:F4} -> {trainer.Evaluate(Data(0)):F6} in {trainer.Steps()} steps");
        Console.WriteLine($"trainer w: {Fmt(trainer.Params()["w"])}");

        var path = Path.Combine(Path.GetTempPath(), "rlx-example-train-cs.gguf");
        var written = trainer.Save(path);
        var restored = new RLX.Trainer(g.Clone(), new[] { "w" }, new Dictionary<string, float[]> { ["w"] = new float[3] }, new { device });
        Console.WriteLine($"checkpoint: {written} tensors, restored step {restored.Load(path)["step"]}");
        File.Delete(path);
        restored.Freeze(new[] { "w" });
        var before = restored.Params()["w"];
        restored.Step(Data(1));
        Console.WriteLine($"frozen: [{string.Join(", ", restored.Frozen())}] unchanged: {restored.Params()["w"].SequenceEqual(before)}");
        restored.Unfreeze();

        var backward = new RLX.Session(device).Compile(RLX.Grad(g, new[] { w })); // outputs [loss, dw]
        var opt = new RLX.Optimizer("sgd", new { lr = 0.2 });
        var parameters = new Dictionary<string, float[]> { ["w"] = new float[3] };
        float lossValue = 0;
        for (ulong step = 0; step < 300; step++)
        {
            backward.SetParam("w", parameters["w"]);
            var feed = Data(step);
            feed["d_output"] = new float[] { 1 };
            var outs = backward.Run(feed);
            lossValue = outs[0][0];
            opt.Step(parameters, new Dictionary<string, float[]> { ["w"] = outs[1] });
        }
        Console.WriteLine($"manual loss: {lossValue:F6} w: {Fmt(parameters["w"])}");
    }
}
