# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Distributed: three ranks as tasks on threads, on loopback; across machines,
# one rank per host with the same `peers`. Collectives, pipelined inference (a
# stage shipped per worker), data-parallel training.
#
#     julia -t 3 --project=crates/bindings/rlx-julia crates/bindings/rlx-julia/examples/distributed.jl

using RLX
using Printf

const WORLD = 3
const PER_RANK = 16
const TMP = mktempdir()

function blob(name, values)
    path = joinpath(TMP, name)
    Base.open(io -> Base.write(io, Float32.(values)), path, "w")
    "file://" * path
end

matvec(x, w) = [max(0f0, sum(x[i] * w[(i-1)*4+j] for i in 1:4)) for j in 1:4]

stage_w = Dict(1 => Float32[(i * 5 % 7) / 7 - 0.3 for i in 0:15], 2 => Float32[(i * 3 % 5) / 5 - 0.2 for i in 0:15])
w_true = Float32[1.5, -2, 0.5]
x_all = Float32[(i * 37 % 23) / 11.5 - 1 for i in 0:PER_RANK*WORLD*3-1]
y_all = Float32[sum(x_all[i*3+k+1] * w_true[k+1] for k in 0:2) for i in 0:PER_RANK*WORLD-1]

spec(rank, uris) = Dict(
    "params" => [Dict("name" => "w", "uri" => uris["w"])],
    "data" => [Dict("input" => "X", "uri" => uris["X"], "elem" => 3, "shard_start" => rank * PER_RANK, "shard_len" => PER_RANK),
               Dict("input" => "y", "uri" => uris["y"], "elem" => 1, "shard_start" => rank * PER_RANK, "shard_len" => PER_RANK)],
    "batch" => 4, "lr_per_epoch" => fill(0.2, 40), "device" => "cpu")

peers = RLX.loopback_peers(WORLD)
workers = [Threads.@spawn begin
    g = RLX.connect(RLX.Group, rank, WORLD; options = Dict("peers" => peers, "device" => "cpu"))
    RLX.all_reduce(g, Float32[rank + 1]; op = "sum")
    RLX.all_gather(g, Float32[rank + 1])
    RLX.broadcast(g, 0, Float32[0])   # the root's length; it overwrites
    RLX.serve_worker(g; options = Dict("steps" => 2))
    Float32.(RLX.serve_trainer(g)["params"]["w"])
end for rank in 1:WORLD-1]

coordinator = Threads.@spawn begin
    g = RLX.connect(RLX.Group, 0, WORLD; options = Dict("peers" => peers, "device" => "cpu"))
    println("mesh: rank $(RLX.rank(g)) of $(RLX.world(g)), leader: $(RLX.is_leader(g))")
    println("all_reduce sum: ", RLX.all_reduce(g, Float32[1]; op = "sum"))
    println("all_gather: ", RLX.all_gather(g, Float32[1]))
    println("broadcast: ", RLX.broadcast(g, 0, Float32[42]))
    caps = RLX.collect_caps(g)
    println("workers: ", join(["rank $(c["rank"]) on $(c["platform"])" for c in caps], ", "))
    stage = RLX.Graph(name = "stage")
    sx = RLX.input(stage, "x", [1, 4])
    RLX.set_outputs(stage, [RLX.relu(stage, RLX.matmul(stage, sx, RLX.param(stage, "w", [4, 4])))])
    for r in 1:WORLD-1
        RLX.ship_stage(g, r, stage, Dict("device" => "cpu", "weights" => Dict("w" => blob("w$r.bin", stage_w[r]))))
    end
    for xv in (Float32[1, -1, 0.5, 2], Float32[0.25, 0.5, -0.75, 1])
        RLX.send_activation(g, 1, xv)
        yv = RLX.recv_activation(g, WORLD - 1)
        @printf("pipeline max|Δ| vs local = %.1e\n", maximum(abs.(yv .- matvec(matvec(xv, stage_w[1]), stage_w[2]))))
    end
    RLX.collect_caps(g)
    lr = RLX.Graph(name = "lr")
    X = RLX.input(lr, "X", [4, 3]); Y = RLX.input(lr, "y", [4, 1]); wv = RLX.param(lr, "w", [3, 1])
    d = RLX.sub(lr, RLX.matmul(lr, X, wv), Y)
    RLX.set_outputs(lr, [RLX.mean(lr, RLX.mul(lr, d, d), [0, 1])])
    bwd = RLX.grad(lr, [wv])
    uris = Dict("X" => blob("X.bin", x_all), "y" => blob("y.bin", y_all), "w" => blob("w0.bin", zeros(Float32, 3)))
    for r in 1:WORLD-1
        RLX.ship_train(g, r, bwd, spec(r, uris))
    end
    mine = RLX.run_train(g, bwd, spec(0, uris))
    m = mine["metrics"]
    @printf("training loss: %.4f -> %.2e\n", m["first_loss"], m["last_loss"])
    Float32.(mine["params"]["w"])
end

w0 = fetch(coordinator)
println("rank 0 w: ", join([@sprintf("%.3f", v) for v in w0], " "), " (true: 1.5 -2 0.5)")
for (i, t) in enumerate(workers)
    println("rank $i w agrees: ", maximum(abs.(fetch(t) .- w0)) < 1e-5)
end
rm(TMP; recursive = true)
