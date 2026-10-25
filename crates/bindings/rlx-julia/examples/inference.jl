# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Inference: build a graph, compile it for a device, set its weights, run it.
#
#     julia --project=crates/bindings/rlx-julia crates/bindings/rlx-julia/examples/inference.jl [device]
#
# A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

using RLX
using Printf

weights(n, seed) = Float32[((i * 7919 + seed * 104729) % 1000) / 1000 - 0.5 for i in 0:n-1]

device = isempty(ARGS) ? "cpu" : ARGS[1]
g = RLX.Graph(name = "classifier")
x = RLX.input(g, "x", [1, 4])
hidden = RLX.relu(g, RLX.matmul(g, x, RLX.param(g, "w1", [4, 8])))
RLX.set_outputs(g, [RLX.softmax(g, RLX.matmul(g, hidden, RLX.param(g, "w2", [8, 3])))])

model = RLX.compile(RLX.Session(device = device), g)   # compile consumes the graph
println("compiled on: ", RLX.device(model))
RLX.set_params(model, Dict("w1" => weights(32, 1), "w2" => weights(24, 2)))
probs = RLX.run(model, Dict("x" => Float32[0.5, -1, 2, 0.25]))[1]
println("probs: ", join([@sprintf("%.4f", p) for p in probs], " "))
@printf("probs sum: %.4f\n", sum(probs))
println("output shapes: ", RLX.output_shapes(model), " dtypes: ", RLX.output_dtypes(model))
