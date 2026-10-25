# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Picking a backend: what this build has, what suits a graph, and the ways to
# choose — by name, by cost model, by policy, by benchmark, per run.
#
#     julia --project=crates/bindings/rlx-julia crates/bindings/rlx-julia/examples/backends.jl

using RLX
using Printf

function model()
    g = RLX.Graph(name = "mlp")
    x = RLX.input(g, "x", [4, 16])
    RLX.set_outputs(g, [RLX.tanh(g, RLX.matmul(g, x, RLX.param(g, "w", [16, 16])))])
    g
end
x = Dict("x" => Float32[(i % 7) / 7 for i in 0:63])
w = Float32[(i % 5) / 10 - 0.2 for i in 0:255]

available = RLX.devices()
println("devices: ", join(available, " "))
println("metal available: ", RLX.is_available("metal"))
println("fastest for graph: ", RLX.fastest_device_for(model()))
reference = nothing
for device in available
    c = RLX.compile(RLX.Session(device = device), model())
    RLX.set_param(c, "w", w)
    y = RLX.run(c, x)[1]
    global reference = something(reference, y)
    @printf("session %s: max|Δ| vs first = %.2e\n", device, maximum(abs.(y .- reference)))
end
picked = RLX.compile(RLX.FlexibleSession(policy = Dict("prefer" => ["metal", "cpu"])), model())
println("flexible session picked: ", RLX.device(picked))
runner = RLX.Runner(model())
RLX.set_param(runner, "w", w)
println("runner warmed: ", join(RLX.warm_all(runner), " "))
for row in RLX.benchmark(runner, x; runs = 3)
    println("  bench $(row["device"]): exec_ms=$(row["exec_ms"])")
end
println("runner ran on: ", RLX.run(runner, x).device)
println("run_try chain: ", RLX.run_try(runner, ["metal", "cpu"], x).device)
router = RLX.Router(model())
RLX.set_param(router, "w", w)
println("router ran on: ", RLX.run_chain(router, x).device)
