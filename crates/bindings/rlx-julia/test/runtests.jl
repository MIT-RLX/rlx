# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# The Julia binding against the real library. Run with threads for the mesh:
#     julia -t 2 --project=crates/bindings/rlx-julia crates/bindings/rlx-julia/test/runtests.jl

using Test
using RLX

"loss = mean((x·w − y)²), x/y [1, 2], w [2, 2]."
function loss_graph()
    g = RLX.Graph(name = "mse")
    x = RLX.input(g, "x", [1, 2]); y = RLX.input(g, "y", [1, 2])
    w = RLX.param(g, "w", [2, 2])
    d = RLX.sub(g, RLX.matmul(g, x, w), y)
    RLX.set_outputs(g, [RLX.mean(g, RLX.mul(g, d, d), [0, 1])])
    g, w
end

const BATCH = Dict("x" => Float32[1, 1], "y" => Float32[1, -1])

@testset "RLX.jl" begin
    @testset "inference" begin
        g, _ = loss_graph()
        m = RLX.compile(RLX.Session(device = "cpu"), g)
        @test RLX.is_consumed(g)
        RLX.set_param(m, "w", Float32[1, 0, 0, 1])
        out = RLX.run(m, Dict("x" => Float32[1, 2], "y" => Float32[0, 0]))
        @test out[1][1] ≈ 2.5
        @test RLX.device(m) == "cpu"
        @test RLX.output_shapes(m) == [[]]
        @test RLX.version() isa String
    end

    @testset "errors" begin
        g = RLX.Graph()
        e = try RLX.relu(g, 42); nothing catch err; err end
        @test e isa RLX.RLXError && e.kind == :range && occursin("does not exist", e.message)
        @test_throws RLX.RLXError RLX.Session(device = "no-such-device")
    end

    @testset "training" begin
        g, w = loss_graph()
        bwd = RLX.compile(RLX.Session(), RLX.grad(g, [w]))
        RLX.set_param(bwd, "w", Float32[1, 0, 0, 1])
        outs = RLX.run(bwd, Dict("x" => Float32[1, 2], "y" => Float32[0, 0], "d_output" => Float32[1]))
        @test outs[2] ≈ Float32[1, 2, 2, 4]

        t = RLX.Trainer(RLX.clone(g), ["w"], Dict("w" => zeros(Float32, 4));
                        options = Dict("optimizer" => Dict("kind" => "adamw", "lr" => 0.1)))
        summary = RLX.run(t, 300; batch = s -> BATCH, on_step = (s, l) -> l > 1e-6)
        @test summary["last_loss"] < 1e-4
        @test RLX.trainable(t) == ["w"]
        path = joinpath(tempdir(), "rlx-jl-ck.gguf")
        @test RLX.save(t, path) >= 1
        t2 = RLX.Trainer(RLX.clone(g), ["w"], Dict("w" => zeros(Float32, 4)))
        @test RLX.load(t2, path)["step"] == summary["steps"]
        rm(path; force = true)
        @test RLX.freeze(t2) == 1

        opt = RLX.Optimizer(kind = "sgd", options = Dict("lr" => 0.5))
        p = Dict("p" => Float32[1, -1])
        RLX.step!(opt, p, Dict("p" => Float32[2, -2]))
        @test p["p"] == Float32[0, 0]
    end

    @testset "loading" begin
        path = joinpath(tempdir(), "rlx-jl.gguf")
        w = Float32[0.5, -1, 2, 0.25]
        writer = RLX.GgufWriter()
        RLX.add_f32(writer, "w", [4], "F32", w)
        @test RLX.write(writer, path; architecture = "demo", metadata = Dict("demo.layers" => 1)) == 1
        f = RLX.open(RLX.Gguf, path)
        @test RLX.architecture(f) == "demo"
        @test RLX.tensor(f, "w") == w
        @test RLX.info(f, "w")["dtype"] == "F32"
        @test RLX.metadata_value(f, "demo.layers") == 1
        rm(path; force = true)
        packed = RLX.quantize(repeat(w, 8), "Q8_0")
        @test length(RLX.dequantize(packed, "Q8_0")) == 32
    end

    @testset "backends" begin
        @test "cpu" in RLX.devices()
        @test RLX.is_available("cpu")
        g, _ = loss_graph()
        @test RLX.fastest_device_for(g) == "cpu"
        r = RLX.Runner(g; policy = Dict("only" => ["cpu"]))
        RLX.set_param(r, "w", Float32[1, 0, 0, 1])
        res = RLX.run(r, Dict("x" => Float32[1, 2], "y" => Float32[0, 0]))
        @test res.device == "cpu" && res.outputs[1][1] ≈ 2.5
        @test RLX.warm_all(r) == ["cpu"]
        @test length(RLX.benchmark(r, Dict("x" => Float32[1, 2], "y" => Float32[0, 0]); runs = 1)) == 1
    end

    @testset "a two-rank mesh" begin
        if Threads.nthreads() < 2
            @info "skipping the mesh: run with `julia -t 2`"
        else
            peers = RLX.loopback_peers(2)
            # Both ranks are tasks, so the main task only waits: a rank blocked in
            # a call never holds the thread its peer needs.
            worker = Threads.@spawn begin
                g = RLX.connect(RLX.Group, 1, 2; options = Dict("peers" => peers))
                RLX.all_reduce(g, Float32[3]; op = "max")
                RLX.serve_worker(g; options = Dict("steps" => 1))
            end
            coordinator = Threads.@spawn begin
                g = RLX.connect(RLX.Group, 0, 2; options = Dict("peers" => peers))
                reduced = RLX.all_reduce(g, Float32[5]; op = "max")
                caps = RLX.collect_caps(g)
                s = RLX.Graph(name = "double")
                x = RLX.input(s, "x", [2])
                RLX.set_outputs(s, [RLX.add(s, x, x)])
                RLX.ship_stage(g, 1, s, Dict("device" => "cpu"))
                RLX.send_activation(g, 1, Float32[1, 2])
                (reduced = reduced, caps = caps, y = RLX.recv_activation(g, 1))
            end
            r = fetch(coordinator)
            @test r.reduced == Float32[5]
            @test length(r.caps) == 1
            @test r.y == Float32[2, 4]
            @test fetch(worker)["activations"] == 1
        end
    end
end
