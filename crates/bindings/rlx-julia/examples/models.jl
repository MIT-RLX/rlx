# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Models: any provider's model, opened by path or id and run by name. A graph
# saved as an .rlxp package runs anywhere; a plugin's models (a downstream collection's LLMs,
# or the reference plugin's demo:count) load through RLX_MODEL_PLUGINS.
#   RLX_MODEL_PLUGINS=target/debug/librlx_model_plugin.dylib julia --project=. examples/models.jl

using RLX
using Printf

g = RLX.Graph(name = "affine")
x = RLX.input(g, "x", [1, 2])
RLX.set_outputs(g, [RLX.matmul(g, x, RLX.param(g, "w", [2, 2]))])
path = joinpath(tempdir(), "rlx-example-julia.rlxp")
RLX.save_model(g, path, Dict("w" => Float32[2, 0, 0, 3]);
               model = Dict("task" => "graph", "family" => "affine", "outputs" => ["y"]))

m = RLX.open(RLX.Model, path; options = Dict("device" => "cpu"))
info = RLX.info(m)
println("model: $(info["family"]) ($(info["task"])) on $(info["device"])")
y = RLX.run(m, Dict("x" => Float32[1, 1]))[1]
@printf("model run: %.3f %.3f\n", y[1], y[2])
rm(path)

lm = RLX.open(RLX.Model, "demo:count")
info = RLX.info(lm)
println("plugin model: $(info["family"]) ($(info["task"]))")
println("model text: ", RLX.generate_text(lm, "c"; options = Dict("max_tokens" => 4)))
println("model ids: ", join(RLX.generate(lm, UInt32[2]; max_tokens = 3), " "))

# By task: a chat, speech out and back in, embeddings.
println("chat: ", RLX.chat(lm, [Dict("role" => "user", "content" => "c")]; options = Dict("max_tokens" => 4)))
speech = RLX.synthesize(RLX.open(RLX.Model, "demo:voice"), "hello")
println("voice: $(length(speech.audio)) samples at $(speech.sample_rate) Hz")
println("heard: ", RLX.transcribe(RLX.open(RLX.Model, "demo:speech"), speech.audio; sample_rate = speech.sample_rate))
rows = RLX.embed(RLX.open(RLX.Model, "demo:embed"), ["hello", "sky"])
println("embed: ", join([Int(v) for row in rows for v in row], " "))
