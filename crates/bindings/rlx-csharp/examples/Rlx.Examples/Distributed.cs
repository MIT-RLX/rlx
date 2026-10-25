// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

using System.Text.Json.Nodes;
using Rlx;

/// Distributed: three ranks as tasks on loopback; across machines, one rank per
/// host with the same peers. Collectives, pipelined inference (a stage shipped
/// per worker), data-parallel training.
static class Distributed
{
    const int World = 3, PerRank = 16;
    static readonly string Tmp = Directory.CreateTempSubdirectory("rlx-dist-cs").FullName;

    static string Blob(string name, float[] values)
    {
        var path = Path.Combine(Tmp, name);
        File.WriteAllBytes(path, values.SelectMany(BitConverter.GetBytes).ToArray());
        return "file://" + path;
    }

    static float[] Matvec(float[] x, float[] w) =>
        Enumerable.Range(0, 4).Select(j => Math.Max(0, Enumerable.Range(0, 4).Sum(i => x[i] * w[i * 4 + j]))).ToArray();

    static object Spec(int rank, Dictionary<string, string> uris) => new
    {
        @params = new[] { new { name = "w", uri = uris["w"] } },
        data = new object[]
        {
            new { input = "X", uri = uris["X"], elem = 3, shard_start = rank * PerRank, shard_len = PerRank },
            new { input = "y", uri = uris["y"], elem = 1, shard_start = rank * PerRank, shard_len = PerRank },
        },
        batch = 4,
        lr_per_epoch = Enumerable.Repeat(0.2, 40).ToArray(),
        device = "cpu",
    };

    static float[] WeightsOf(JsonObject report) =>
        report["params"]!["w"]!.AsArray().Select(v => v!.GetValue<float>()).ToArray();

    public static void Run()
    {
        var stageW = new Dictionary<int, float[]>
        {
            [1] = Enumerable.Range(0, 16).Select(i => (i * 5 % 7) / 7f - 0.3f).ToArray(),
            [2] = Enumerable.Range(0, 16).Select(i => (i * 3 % 5) / 5f - 0.2f).ToArray(),
        };
        var wTrue = new[] { 1.5f, -2, 0.5f };
        var xAll = Enumerable.Range(0, PerRank * World * 3).Select(i => (i * 37 % 23) / 11.5f - 1).ToArray();
        var yAll = Enumerable.Range(0, PerRank * World).Select(i => Enumerable.Range(0, 3).Sum(k => xAll[i * 3 + k] * wTrue[k])).ToArray();

        var peers = RLX.LoopbackPeers(World);
        var workers = Enumerable.Range(1, World - 1).Select(rank => Task.Run(() =>
        {
            var g = RLX.Group.Connect((uint)rank, World, new { peers, device = "cpu" });
            g.AllReduce(new float[] { rank + 1 }, "sum");
            g.AllGather(new float[] { rank + 1 });
            g.Broadcast(0, new float[] { 0 }); // the root's length; it overwrites
            g.ServeWorker(new { steps = 2 });
            return WeightsOf(g.ServeTrainer());
        })).ToList();

        var mesh = RLX.Group.Connect(0, World, new { peers, device = "cpu" });
        Console.WriteLine($"mesh: rank {mesh.Rank()} of {mesh.World()}, leader: {mesh.IsLeader()}");
        Console.WriteLine($"all_reduce sum: [{string.Join(", ", mesh.AllReduce(new float[] { 1 }, "sum"))}]");
        Console.WriteLine($"all_gather: [{string.Join(", ", mesh.AllGather(new float[] { 1 }))}]");
        Console.WriteLine($"broadcast: [{string.Join(", ", mesh.Broadcast(0, new float[] { 42 }))}]");
        var caps = mesh.CollectCaps();
        Console.WriteLine("workers: " + string.Join(", ", caps.Select(c => $"rank {c!["rank"]} on {c["platform"]}")));
        var stage = new RLX.Graph("stage");
        var sx = stage.Input("x", new long[] { 1, 4 });
        stage.SetOutputs(new[] { stage.Relu(stage.Matmul(sx, stage.Param("w", new long[] { 4, 4 }))) });
        for (int r = 1; r < World; r++)
            mesh.ShipStage((uint)r, stage, new { device = "cpu", weights = new { w = Blob($"w{r}.bin", stageW[r]) } });
        foreach (var xv in new[] { new[] { 1f, -1, 0.5f, 2 }, new[] { 0.25f, 0.5f, -0.75f, 1 } })
        {
            mesh.SendActivation(1, xv);
            var yv = mesh.RecvActivation(World - 1);
            var want = Matvec(Matvec(xv, stageW[1]), stageW[2]);
            Console.WriteLine($"pipeline max|Δ| vs local = {yv.Zip(want, (a, b) => Math.Abs(a - b)).Max():0.0e+00}");
        }

        mesh.CollectCaps();
        var lr = new RLX.Graph("lr");
        var X = lr.Input("X", new long[] { 4, 3 });
        var Y = lr.Input("y", new long[] { 4, 1 });
        var wv = lr.Param("w", new long[] { 3, 1 });
        var d = lr.Sub(lr.Matmul(X, wv), Y);
        lr.SetOutputs(new[] { lr.Mean(lr.Mul(d, d), new long[] { 0, 1 }) });
        var bwd = RLX.Grad(lr, new[] { wv });
        var uris = new Dictionary<string, string> { ["X"] = Blob("X.bin", xAll), ["y"] = Blob("y.bin", yAll), ["w"] = Blob("w0.bin", new float[3]) };
        for (int r = 1; r < World; r++) mesh.ShipTrain((uint)r, bwd, Spec(r, uris));
        var mine = mesh.RunTrain(bwd, Spec(0, uris));
        Console.WriteLine($"training loss: {mine["metrics"]!["first_loss"]!.GetValue<double>():F4} -> {mine["metrics"]!["last_loss"]!.GetValue<double>():0.00e+00}");
        var w0 = WeightsOf(mine);
        Console.WriteLine($"rank 0 w: {string.Join(" ", w0.Select(v => v.ToString("F3")))} (true: 1.5 -2 0.5)");
        for (int r = 1; r < World; r++)
        {
            var theirs = workers[r - 1].Result;
            Console.WriteLine($"rank {r} w agrees: {theirs.Zip(w0, (a, b) => Math.Abs(a - b) < 1e-5).All(t => t)}");
        }
        Directory.Delete(Tmp, true);
    }
}
