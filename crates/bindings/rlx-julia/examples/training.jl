# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
# the same fit by hand with `grad` and an Optimizer.
#
#     julia --project=crates/bindings/rlx-julia crates/bindings/rlx-julia/examples/training.jl [device]

using RLX
using Printf

const W_TRUE = Float32[2, -1, 0.5]
const BATCH = 8

function data(step)
    x = Float32[]; y = Float32[]
    for i in 0:BATCH-1
        row = Float32[((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1 for k in 0:2]
        append!(x, row); push!(y, sum(row .* W_TRUE))
    end
    Dict("x" => x, "y" => y)
end

fmt(v) = join([@sprintf("%.3f", a) for a in v], " ")
device = isempty(ARGS) ? "cpu" : ARGS[1]
g = RLX.Graph(name = "regression")
x = RLX.input(g, "x", [BATCH, 3]); y = RLX.input(g, "y", [BATCH, 1])
w = RLX.param(g, "w", [3, 1])
d = RLX.sub(g, RLX.matmul(g, x, w), y)
RLX.set_outputs(g, [RLX.mean(g, RLX.mul(g, d, d), [0, 1])])

trainer = RLX.Trainer(RLX.clone(g), ["w"], Dict("w" => zeros(Float32, 3));
    options = Dict("device" => device, "optimizer" => Dict("kind" => "adamw", "lr" => 0.1)))
first = RLX.evaluate(trainer, data(0))
RLX.run(trainer, 400; batch = data, lr = s -> 0.1 * 0.5 * (1 + cos(pi * s / 400)), on_step = (s, l) -> l > 1e-6)
@printf("trainer loss: %.4f -> %.6f in %d steps\n", first, RLX.evaluate(trainer, data(0)), RLX.steps(trainer))
println("trainer w: ", fmt(RLX.params(trainer)["w"]))

path = joinpath(tempdir(), "rlx-example-train-jl.gguf")
written = RLX.save(trainer, path)
restored = RLX.Trainer(RLX.clone(g), ["w"], Dict("w" => zeros(Float32, 3)); options = Dict("device" => device))
println("checkpoint: $written tensors, restored step ", RLX.load(restored, path)["step"])
rm(path)
RLX.freeze(restored; names = ["w"])
before = copy(RLX.params(restored)["w"])
RLX.step(restored, data(1))
println("frozen: ", RLX.frozen(restored), " unchanged: ", RLX.params(restored)["w"] == before)
RLX.unfreeze(restored)

backward = RLX.compile(RLX.Session(device = device), RLX.grad(g, [w]))   # outputs [loss, dw]
opt = RLX.Optimizer(kind = "sgd", options = Dict("lr" => 0.2))
params = Dict("w" => zeros(Float32, 3))
loss = 0f0
for step in 0:299
    RLX.set_param(backward, "w", params["w"])
    outs = RLX.run(backward, merge(data(step), Dict("d_output" => Float32[1])))
    global loss = outs[1][1]
    RLX.step!(opt, params, Dict("w" => outs[2]))
end
@printf("manual loss: %.6f w: %s\n", loss, fmt(params["w"]))
