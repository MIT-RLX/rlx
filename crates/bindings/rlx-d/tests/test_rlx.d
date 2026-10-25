// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The D binding against the real library; exits non-zero on the first failure.
module test_rlx;

import core.stdc.stdlib : exit;
import std.file : remove, tempDir;
import std.path : buildPath;
import std.stdio : stderr, writefln;
import rlx;

int checks;

void check(bool ok, string what)
{
    checks++;
    if (!ok)
    {
        stderr.writeln("failed: ", what);
        exit(1);
    }
}

Graph lossGraph(out uint w)
{
    auto g = new Graph(name: "mse");
    immutable x = g.input("x", [1, 2]);
    immutable y = g.input("y", [1, 2]);
    w = g.param("w", [2, 2]);
    immutable d = g.sub(g.matmul(x, w), y);
    g.setOutputs([g.mean(g.mul(d, d), [0, 1])]);
    return g;
}

void main()
{
    uint w;
    {
        auto g = lossGraph(w);
        auto m = new Session().compile(g);
        check(g.isConsumed(), "compile consumes");
        m.setParam("w", [1, 0, 0, 1]);
        check(m.run(["x": [1.0f, 2], "y": [0.0f, 0]])[0][0] == 2.5, "loss 2.5");
        check(m.device() == "cpu", "device");
    }
    try
    {
        new Graph().relu(42);
        check(false, "range error raised");
    }
    catch (RlxException e)
        check(e.kind == RlxErrorKind.range, "range error");
    {
        auto g = lossGraph(w);
        auto bwd = new Session().compile(grad(g, [w]));
        bwd.setParam("w", [1, 0, 0, 1]);
        check(bwd.run(["x": [1.0f, 2], "y": [0.0f, 0], "d_output": [1.0f]])[1][3] == 4, "dw");
        auto t = new Trainer(g.clone(), ["w"], ["w": [0.0f, 0, 0, 0]],
            options: parseJSON(`{"optimizer": {"kind": "adamw", "lr": 0.1}}`));
        const(float[])[string] b = ["x": [1.0f, 1], "y": [1.0f, -1]];
        auto r = t.run(300, (s) => b, onStep: (s, l) => l > 1e-6);
        check(r.lastLoss.get < 1e-4, "trainer converges");
        auto opt = new Optimizer(kind: "sgd", options: parseJSON(`{"lr": 0.5}`));
        float[][string] p = ["p": [1.0f, -1]];
        opt.step(p, ["p": [2.0f, -2]]);
        check(p["p"] == [0.0f, 0], "optimizer in place");
    }
    {
        immutable path = buildPath(tempDir, "rlx-d-test.gguf");
        auto gw = new GgufWriter();
        gw.addF32("w", [4], "F32", [0.5, -1, 2, 0.25]);
        check(gw.write(path, architecture: "demo") == 1, "write");
        auto f = Gguf.open(path);
        check(f.architecture() == "demo", "architecture");
        check(f.tensor("w")[2] == 2, "tensor");
        check(f.info("w")["dtype"].str == "F32", "info");
        remove(path);
    }
    {
        check(devices()[0] == "cpu" && isAvailable("cpu"), "cpu");
        auto g = lossGraph(w);
        auto r = new Runner(g, policy: parseJSON(`{"only": ["cpu"]}`));
        r.setParam("w", [1, 0, 0, 1]);
        auto res = r.run(["x": [1.0f, 2], "y": [0.0f, 0]]);
        check(res.device == "cpu" && res.outputs[0][0] == 2.5, "runner");
        check(Group.connect(0, 1).allReduce([2.0f, 3], op: "max")[1] == 3, "all_reduce");
    }
    writefln("test_rlx: %d checks passed", checks);
}
