# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Inference: build a graph, compile it for a device, set its weights, run it.
#   mix run examples/inference.exs [device]
# A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
alias Rlx.{Compiled, Graph, Session}

device = List.first(System.argv()) || "cpu"
weights = fn n, seed -> for i <- 0..(n - 1), do: rem(i * 7919 + seed * 104_729, 1000) / 1000 - 0.5 end

g = Graph.new(name: "classifier")
x = Graph.input(g, "x", [1, 4])
hidden = Graph.relu(g, Graph.matmul(g, x, Graph.param(g, "w1", [4, 8])))
Graph.set_outputs(g, [Graph.softmax(g, Graph.matmul(g, hidden, Graph.param(g, "w2", [8, 3])))])

model = Session.compile(Session.new(device: device), g)
IO.puts("compiled on: #{Compiled.device(model)}")
Compiled.set_params(model, %{"w1" => weights.(32, 1), "w2" => weights.(24, 2)})
[probs] = Compiled.run(model, %{"x" => [0.5, -1, 2, 0.25]})
IO.puts("probs: #{Enum.map_join(probs, " ", &:erlang.float_to_binary(&1, decimals: 4))}")
IO.puts("probs sum: #{:erlang.float_to_binary(Enum.sum(probs), decimals: 4)}")
IO.puts("output shapes: #{inspect(Compiled.output_shapes(model))} dtypes: #{inspect(Compiled.output_dtypes(model))}")
