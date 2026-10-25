// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: each rank is a process; start one per rank with the same peers
// (here three on loopback; across machines, one per host):
//   ./distributed 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   ./distributed 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   ./distributed 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
// (build: ldc2 -i -Isource examples/distributed.d -L-lrlx_capi). Rank 0
// coordinates: collectives, pipelined inference (a stage shipped per worker),
// data-parallel training. With no arguments it is a mesh of one.
module distributed;

import std.algorithm : map;
import std.array : array, join, split;
import std.conv : to;
import std.file : remove, tempDir, write;
import std.format : format;
import std.json : JSONValue;
import std.math : abs;
import std.path : buildPath;
import std.range : repeat;
import std.stdio : writefln, writeln;
import rlx;

enum perRank = 16;

void main(string[] args)
{
    immutable rank = args.length > 1 ? args[1].to!uint : 0;
    immutable world = args.length > 2 ? args[2].to!uint : 1;
    string[] blobs;
    string blob(string name, const(float)[] values)
    {
        immutable path = buildPath(tempDir, format("rlx-dist-d-%d-%s", rank, name));
        write(path, cast(const(ubyte)[]) values);
        blobs ~= path;
        return "file://" ~ path;
    }

    JSONValue options = ["device": "cpu"];
    if (args.length > 3)
        options["peers"] = args[3].split(",");
    auto g = Group.connect(rank, world, options: options);
    auto total = g.allReduce([rank + 1.0f], op: "sum");
    auto gathered = g.allGather([rank + 1.0f]);
    auto shared_ = g.broadcast(0, [rank == 0 ? 42.0f : 0]); // the root's length
    if (rank == 0)
    {
        writefln("mesh: rank %d of %d, leader: %s", g.rank(), g.world(), g.isLeader());
        writeln("all_reduce sum: ", total.map!(v => v.to!string).join(","));
        writeln("all_gather: ", gathered.map!(v => v.to!string).join(","));
        writeln("broadcast: ", shared_.map!(v => v.to!string).join(","));
    }
    if (world == 1)
    {
        writeln("a mesh of one: pass rank world peers to start more ranks");
        return;
    }
    if (rank != 0)
    {
        g.serveWorker(options: parseJSON(`{"steps": 2}`));
        auto w = g.serveTrainer()["params"]["w"].array.map!(v => cast(float) v.floating).array;
        g.send(0, 1, w); // rank 0 compares
        return;
    }

    float[][2] stageW;
    foreach (i; 0 .. 16)
    {
        stageW[0] ~= (i * 5 % 7) / 7.0f - 0.3f;
        stageW[1] ~= (i * 3 % 5) / 5.0f - 0.2f;
    }
    string[] caps;
    foreach (c; g.collectCaps().array)
        caps ~= format("rank %d on %s", c["rank"].integer, c["platform"].str);
    writeln("workers: ", caps.join(", "));
    auto stage = new Graph(name: "stage");
    stage.setOutputs([stage.relu(stage.matmul(stage.input("x", [1, 4]), stage.param("w", [4, 4])))]);
    foreach (r; 1 .. world)
    {
        JSONValue spec = ["device": "cpu"];
        spec["weights"] = ["w": blob(format("w%d.bin", r), stageW[(r - 1) % 2])];
        g.shipStage(r, stage, spec);
    }
    const float[][] xs = [[1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]];
    foreach (xv; xs)
    {
        g.sendActivation(1, xv);
        writeln("pipeline: ", g.recvActivation(world - 1).map!(v => format("%.3f", v)).join(" "));
    }

    g.collectCaps();
    auto lr = new Graph(name: "lr");
    immutable bigX = lr.input("X", [4, 3]);
    immutable y = lr.input("y", [4, 1]);
    immutable wv = lr.param("w", [3, 1]);
    immutable d = lr.sub(lr.matmul(bigX, wv), y);
    lr.setOutputs([lr.mean(lr.mul(d, d), [0, 1])]);
    auto bwd = grad(lr, [wv]);
    immutable float[3] wTrue = [1.5, -2, 0.5];
    immutable n = perRank * world;
    float[] xAll, yAll;
    foreach (i; 0 .. n * 3)
        xAll ~= (i * 37 % 23) / 11.5f - 1;
    foreach (i; 0 .. n)
        yAll ~= xAll[i * 3] * wTrue[0] + xAll[i * 3 + 1] * wTrue[1] + xAll[i * 3 + 2] * wTrue[2];
    immutable ux = blob("X.bin", xAll), uy = blob("y.bin", yAll), uw = blob("w0.bin", [0.0f, 0, 0]);
    JSONValue spec(uint r)
    {
        return parseJSON(format(`{"params": [{"name": "w", "uri": "%s"}],
            "data": [{"input": "X", "uri": "%s", "elem": 3, "shard_start": %d, "shard_len": %d},
                     {"input": "y", "uri": "%s", "elem": 1, "shard_start": %d, "shard_len": %d}],
            "batch": 4, "lr_per_epoch": [%s], "device": "cpu"}`,
            uw, ux, r * perRank, perRank, uy, r * perRank, perRank, "0.2".repeat(40).join(",")));
    }
    foreach (r; 1 .. world)
        g.shipTrain(r, bwd, spec(r));
    auto mine = g.runTrain(bwd, spec(0));
    writefln("training loss: %.4f -> %.2e", mine["metrics"]["first_loss"].floating, mine["metrics"]["last_loss"].floating);
    auto w0 = mine["params"]["w"].array.map!(v => cast(float) v.floating).array;
    writeln("rank 0 w: ", w0.map!(v => format("%.3f", v)).join(" "), " (true: 1.5 -2 0.5)");
    foreach (r; 1 .. world)
    {
        auto theirs = g.recv(r, 1);
        bool agree = true;
        foreach (k; 0 .. w0.length)
            agree = agree && abs(theirs[k] - w0[k]) < 1e-5;
        writefln("rank %d w agrees: %s", r, agree);
    }
    foreach (p; blobs)
        remove(p);
}
