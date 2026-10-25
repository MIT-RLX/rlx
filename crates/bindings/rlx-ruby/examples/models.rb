# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Models: any provider's model, opened by path or id and run by name. A graph
# saved as an .rlxp package runs anywhere; a plugin's models (a downstream collection's LLMs,
# or the reference plugin's demo:count) load through RLX_MODEL_PLUGINS.
#   RLX_MODEL_PLUGINS=target/debug/librlx_model_plugin.dylib ruby -Ilib examples/models.rb

require "rlx"
require "tmpdir"

g = RLX::Graph.new(name: "affine")
x = g.input("x", [1, 2])
g.set_outputs([g.matmul(x, g.param("w", [2, 2]))])
path = File.join(Dir.tmpdir, "rlx-example-ruby.rlxp")
RLX.save_model(g, path, { "w" => [2, 0, 0, 3] }, model: { task: "graph", family: "affine", outputs: ["y"] })

m = RLX::Model.open(path, options: { device: "cpu" })
info = m.info
puts "model: #{info["family"]} (#{info["task"]}) on #{info["device"]}"
y = m.run({ "x" => [1, 1] })[0]
puts format("model run: %.3f %.3f", y[0], y[1])
File.delete(path)

lm = RLX::Model.open("demo:count")
info = lm.info
puts "plugin model: #{info["family"]} (#{info["task"]})"
puts "model text: #{lm.generate_text("c", options: { max_tokens: 4 })}"
puts "model ids: #{lm.generate([2], max_tokens: 3).join(" ")}"

# By task: a chat, speech out and back in, embeddings.
puts "chat: #{lm.chat([{ role: "user", content: "c" }], options: { max_tokens: 4 })}"
speech = RLX::Model.open("demo:voice").synthesize("hello")
puts "voice: #{speech[:audio].length} samples at #{speech[:sample_rate]} Hz"
puts "heard: #{RLX::Model.open("demo:speech").transcribe(speech[:audio], sample_rate: speech[:sample_rate])}"
rows = RLX::Model.open("demo:embed").embed(["hello", "sky"])
puts "embed: #{rows.flatten.map(&:to_i).join(" ")}"
