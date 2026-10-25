// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with grad and an Optimizer.
//   ldc2 -i -Isource examples/training.d -L-lrlx_capi && ./training [device]
module training;

import std.algorithm : map;
import std.array : join;
import std.file : remove, tempDir;
import std.format : format;
import std.math : PI, cos;
import std.path : buildPath;
import std.stdio : writefln, writeln;
import rlx;

immutable float[3] wTrue = [2, -1, 0.5];
enum batch = 8;

const(float[])[string] data(size_t step, bool seed = false)
{
    float[] x, y;
    foreach (i; 0 .. batch)
    {
        float s = 0;
        foreach (k; 0 .. 3)
        {
            immutable v = ((step * 31 + i * 7 + k * 3) % 17) / 8.5f - 1;
            x ~= v;
            s += v * wTrue[k];
        }
        y ~= s;
    }
    if (seed) // the backward graph's d_output seed
        return ["x": x, "y": y, "d_output": [1.0f]];
    return ["x": x, "y": y];
}

string fmt(const(float)[] v)
{
    return v.map!(e => format("%.3f", e)).join(" ");
}

void main(string[] args)
{
    immutable device = args.length > 1 ? args[1] : "cpu";
    auto g = new Graph(name: "regression");
    immutable x = g.input("x", [batch, 3]);
    immutable y = g.input("y", [batch, 1]);
    immutable w = g.param("w", [3, 1]);
    immutable d = g.sub(g.matmul(x, w), y);
    g.setOutputs([g.mean(g.mul(d, d), [0, 1])]);

    auto trainer = new Trainer(g.clone(), ["w"], ["w": [0.0f, 0, 0]],
        options: parseJSON(`{"device": "` ~ device ~ `", "optimizer": {"kind": "adamw", "lr": 0.1}}`));
    immutable first = trainer.evaluate(data(0));
    trainer.run(400, (s) => data(s), lr: (s) => 0.1 * 0.5 * (1 + cos(PI * s / 400)),
        onStep: (s, loss) => loss > 1e-6);
    writefln("trainer loss: %.4f -> %.6f in %d steps", first, trainer.evaluate(data(0)), trainer.steps());
    writeln("trainer w: ", fmt(trainer.params()["w"]));

    immutable path = buildPath(tempDir, "rlx-example-train-d.gguf");
    immutable written = trainer.save(path);
    auto restored = new Trainer(g.clone(), ["w"], ["w": [0.0f, 0, 0]],
        options: parseJSON(`{"device": "` ~ device ~ `"}`));
    writefln("checkpoint: %d tensors, restored step %d", written, restored.load(path)["step"].integer);
    remove(path);
    restored.freeze(names: ["w"]);
    immutable before = fmt(restored.params()["w"]);
    restored.step(data(1));
    writeln("frozen: ", restored.frozen(), " unchanged: ", fmt(restored.params()["w"]) == before);
    restored.unfreeze();

    auto backward = new Session(device: device).compile(grad(g, [w])); // outputs [loss, dw]
    auto opt = new Optimizer(kind: "sgd", options: parseJSON(`{"lr": 0.2}`));
    float[][string] params = ["w": [0.0f, 0, 0]];
    float loss = 0;
    foreach (step; 0 .. 300)
    {
        backward.setParam("w", params["w"]);
        auto outs = backward.run(data(step, true));
        loss = outs[0][0];
        opt.step(params, ["w": outs[1]]);
    }
    writefln("manual loss: %.6f w: %s", loss, fmt(params["w"]));
}
