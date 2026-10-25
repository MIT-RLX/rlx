# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Models: any provider's model, opened by path or id and run by name. A graph
# saved as an .rlxp package runs anywhere; a plugin's models (a downstream
# collection's LLMs, or the reference plugin's demo:count) load through
# RLX_MODEL_PLUGINS.
#   RLX_MODEL_PLUGINS=…/librlx_model_plugin.so mix run examples/models.exs
alias Rlx.{Graph, Model}

g = Graph.new(name: "affine")
x = Graph.input(g, "x", [1, 2])
Graph.set_outputs(g, [Graph.matmul(g, x, Graph.param(g, "w", [2, 2]))])
path = Path.join(System.tmp_dir!(), "rlx-example-elixir.rlxp")
Rlx.save_model(g, path, %{"w" => [2, 0, 0, 3]}, model: %{task: "graph", family: "affine", outputs: ["y"]})

m = Model.open(path, options: %{device: "cpu"})
info = Model.info(m)
IO.puts("model: #{info["family"]} (#{info["task"]}) on #{info["device"]}")
[y] = Model.run(m, %{"x" => [1, 1]})
IO.puts("model run: #{Enum.map_join(y, " ", &:erlang.float_to_binary(&1 * 1.0, decimals: 3))}")
File.rm!(path)

lm = Model.open("demo:count")
lm_info = Model.info(lm)
IO.puts("plugin model: #{lm_info["family"]} (#{lm_info["task"]})")
IO.puts("model text: #{Model.generate_text(lm, "c", options: %{max_tokens: 4})}")
IO.puts("model ids: #{Enum.join(Model.generate(lm, [2], max_tokens: 3), " ")}")

# By task: a chat, speech out and back in, embeddings.
IO.puts("chat: #{Model.chat(lm, [%{role: "user", content: "c"}], options: %{max_tokens: 4})}")
%{audio: audio, sample_rate: rate} = Model.synthesize(Model.open("demo:voice"), "hello")
IO.puts("voice: #{length(audio)} samples at #{rate} Hz")
IO.puts("heard: #{Model.transcribe(Model.open("demo:speech"), audio, sample_rate: rate)}")
rows = Model.embed(Model.open("demo:embed"), ["hello", "sky"])
IO.puts("embed: #{rows |> List.flatten() |> Enum.map_join(" ", &trunc/1)}")
