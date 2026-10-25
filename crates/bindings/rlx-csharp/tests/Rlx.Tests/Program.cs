// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

using Rlx;

int checks = 0;
void Check(bool ok, string what)
{
    checks++;
    if (!ok) throw new Exception("failed: " + what);
}

// loss = mean((x·w − y)²), x/y [1, 2], w [2, 2].
(RLX.Graph, uint) LossGraph()
{
    var g = new RLX.Graph("mse");
    var x = g.Input("x", new long[] { 1, 2 });
    var y = g.Input("y", new long[] { 1, 2 });
    var w = g.Param("w", new long[] { 2, 2 });
    var d = g.Sub(g.Matmul(x, w), y);
    g.SetOutputs(new[] { g.Mean(g.Mul(d, d), new long[] { 0, 1 }) });
    return (g, w);
}

// inference
var (g, w) = LossGraph();
using (var m = new RLX.Session().Compile(g))
{
    Check(g.IsConsumed(), "compile consumes");
    m.SetParam("w", new float[] { 1, 0, 0, 1 });
    Check(m.Run(new Dictionary<string, float[]> { ["x"] = new float[] { 1, 2 }, ["y"] = new float[] { 0, 0 } })[0][0] == 2.5f, "loss 2.5");
    Check(m.Device() == "cpu", "device");
}

// errors
try { new RLX.Graph().Relu(42); Check(false, "bad node throws"); }
catch (RlxException e) { Check(e.Kind == RlxException.ErrorKind.Range && e.Message.Contains("does not exist"), "range error"); }

// training
(g, w) = LossGraph();
var bwd = new RLX.Session().Compile(RLX.Grad(g, new[] { w }));
bwd.SetParam("w", new float[] { 1, 0, 0, 1 });
var outs = bwd.Run(new Dictionary<string, float[]> { ["x"] = new float[] { 1, 2 }, ["y"] = new float[] { 0, 0 }, ["d_output"] = new float[] { 1 } });
Check(outs[1][3] == 4f, "dw");
var trainer = new RLX.Trainer(g.Clone(), new[] { "w" }, new Dictionary<string, float[]> { ["w"] = new float[4] },
    new { optimizer = new { kind = "adamw", lr = 0.1 } });
var batch = new Dictionary<string, float[]> { ["x"] = new float[] { 1, 1 }, ["y"] = new float[] { 1, -1 } };
var summary = trainer.Run(300, _ => batch, onStep: (_, l) => l > 1e-6f);
Check((float)summary["last_loss"]! < 1e-4f, "trainer converges");
var opt = new RLX.Optimizer("sgd", new { lr = 0.5 });
var p = new float[] { 1, -1 };
opt.Step(new Dictionary<string, float[]> { ["p"] = p }, new Dictionary<string, float[]> { ["p"] = new float[] { 2, -2 } });
Check(p[0] == 0 && p[1] == 0, "optimizer in place");

// loading
var path = Path.Combine(Path.GetTempPath(), "rlx-cs.gguf");
var wv = new float[] { 0.5f, -1, 2, 0.25f };
using (var wr = new RLX.GgufWriter())
{
    wr.AddF32("w", new long[] { 4 }, "F32", wv);
    Check(wr.Write(path, architecture: "demo") == 1, "write");
}
using (var f = RLX.Gguf.Open(path))
{
    Check(f.Architecture() == "demo", "architecture");
    Check(f.Tensor("w").SequenceEqual(wv), "tensor");
    Check((string)f.Info("w")["dtype"]! == "F32", "info");
}
File.Delete(path);

// backends
Check(RLX.Devices().Contains("cpu") && RLX.IsAvailable("cpu"), "cpu");
(g, _) = LossGraph();
var r = new RLX.Runner(g, new { only = new[] { "cpu" } });
r.SetParam("w", new float[] { 1, 0, 0, 1 });
var res = r.Run(new Dictionary<string, float[]> { ["x"] = new float[] { 1, 2 }, ["y"] = new float[] { 0, 0 } });
Check(res.Device == "cpu" && res.Outputs[0][0] == 2.5f, "runner");

// a two-rank mesh, the worker on another thread
var peers = RLX.LoopbackPeers(2);
var worker = Task.Run(() =>
{
    var wg = RLX.Group.Connect(1, 2, new { peers });
    wg.AllReduce(new float[] { 3 }, "max");
    return wg.ServeWorker(new { steps = 1 });
});
var mesh = RLX.Group.Connect(0, 2, new { peers });
Check(mesh.AllReduce(new float[] { 5 }, "max")[0] == 5f, "all_reduce");
Check(mesh.CollectCaps().Count == 1, "caps");
var s = new RLX.Graph();
var sx = s.Input("x", new long[] { 2 });
s.SetOutputs(new[] { s.Add(sx, sx) });
mesh.ShipStage(1, s, new { device = "cpu" });
mesh.SendActivation(1, new float[] { 1, 2 });
var yv = mesh.RecvActivation(1);
Check(yv[0] == 2 && yv[1] == 4, "activation");
Check((int)worker.Result["activations"]! == 1, "worker served");

Console.WriteLine($"Rlx.Tests: {checks} checks passed");
