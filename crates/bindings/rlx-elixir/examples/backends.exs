# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Picking a backend: what this build has, what suits a graph, and the ways to
# choose — by name, by cost model, by policy, by benchmark, per run.
#   mix run examples/backends.exs
alias Rlx.{Compiled, FlexibleSession, Graph, Router, Runner, Session}

model = fn ->
  g = Graph.new(name: "mlp")
  x = Graph.input(g, "x", [4, 16])
  Graph.set_outputs(g, [Graph.tanh(g, Graph.matmul(g, x, Graph.param(g, "w", [16, 16])))])
  g
end

w = for i <- 0..255, do: rem(i, 5) / 10 - 0.2
x = %{"x" => for(i <- 0..63, do: rem(i, 7) / 7)}

available = Rlx.devices()
IO.puts("devices: #{Enum.join(available, " ")}")
IO.puts("metal available: #{Rlx.is_available("metal")}")
IO.puts("fastest for graph: #{Rlx.fastest_device_for(model.())}")

Enum.reduce(available, nil, fn device, reference ->
  c = Session.compile(Session.new(device: device), model.())
  Compiled.set_param(c, "w", w)
  [y] = Compiled.run(c, x)
  reference = reference || y
  d = Enum.zip_with(y, reference, &abs(&1 - &2)) |> Enum.max()
  IO.puts("session #{device}: max|Δ| vs first = #{:io_lib.format("~.2e", [d * 1.0])}")
  reference
end)

flex = FlexibleSession.new(policy: %{prefer: ["metal", "cpu"]})
IO.puts("flexible session picked: #{Compiled.device(FlexibleSession.compile(flex, model.()))}")
runner = Runner.new(model.())
Runner.set_param(runner, "w", w)
IO.puts("runner warmed: #{Enum.join(Runner.warm_all(runner), " ")}")
for row <- Runner.benchmark(runner, x, runs: 3), do: IO.puts("  bench #{row["device"]}: exec_ms=#{row["exec_ms"]}")
IO.puts("runner ran on: #{Runner.run(runner, x).device}")
IO.puts("run_try chain: #{Runner.run_try(runner, ["metal", "cpu"], x).device}")
router = Router.new(model.())
Router.set_param(router, "w", w)
IO.puts("router ran on: #{Router.run_chain(router, x).device}")
