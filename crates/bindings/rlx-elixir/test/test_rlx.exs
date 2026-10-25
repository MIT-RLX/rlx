# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# The Elixir binding against the real library; exits non-zero on the first
# failure.
alias Rlx.{Compiled, Gguf, GgufWriter, Graph, Group, Optimizer, Runner, Session, Trainer}

defmodule Check do
  def start, do: :counters.new(1, [])

  def check(c, ok, what) do
    :counters.add(c, 1, 1)
    unless ok, do: (IO.puts(:stderr, "failed: #{what}"); System.halt(1))
  end
end

c = Check.start()

loss_graph = fn ->
  g = Graph.new(name: "mse")
  x = Graph.input(g, "x", [1, 2])
  y = Graph.input(g, "y", [1, 2])
  w = Graph.param(g, "w", [2, 2])
  d = Graph.sub(g, Graph.matmul(g, x, w), y)
  Graph.set_outputs(g, [Graph.mean(g, Graph.mul(g, d, d), [0, 1])])
  {g, w}
end

feed = %{"x" => [1, 2], "y" => [0, 0]}
{g, _} = loss_graph.()
m = Session.compile(Session.new(), g)
Check.check(c, Graph.is_consumed(g), "compile consumes")
Compiled.set_param(m, "w", [1, 0, 0, 1])
Check.check(c, Compiled.run(m, feed) == [[2.5]], "loss 2.5")
Check.check(c, Compiled.device(m) == "cpu", "device")

e = try do
  Graph.relu(Graph.new(), 42)
rescue
  e in Rlx.Error -> e
end

Check.check(c, match?(%Rlx.Error{kind: :range}, e), "range error")

{g, w} = loss_graph.()
bwd = Session.compile(Session.new(), Rlx.grad(g, [w]))
Compiled.set_param(bwd, "w", [1, 0, 0, 1])
[_, dw] = Compiled.run(bwd, Map.put(feed, "d_output", [1]))
Check.check(c, Enum.at(dw, 3) == 4.0, "dw")
t = Trainer.new(Graph.clone(g), ["w"], %{"w" => [0, 0, 0, 0]}, options: %{optimizer: %{kind: "adamw", lr: 0.1}})
r = Trainer.run(t, 300, fn _ -> %{"x" => [1, 1], "y" => [1, -1]} end, on_step: fn _, l -> l > 1.0e-6 end)
Check.check(c, r.last_loss < 1.0e-4, "trainer converges")
opt = Optimizer.new(kind: "sgd", options: %{lr: 0.5})
Check.check(c, Optimizer.step(opt, %{"p" => [1, -1]}, %{"p" => [2, -2]}) == %{"p" => [0.0, 0.0]}, "optimizer")

path = Path.join(System.tmp_dir!(), "rlx-elixir-test.gguf")
gw = GgufWriter.new()
GgufWriter.add_f32(gw, "w", [4], "F32", [0.5, -1, 2, 0.25])
Check.check(c, GgufWriter.write(gw, path, architecture: "demo") == 1, "write")
f = Gguf.open(path)
Check.check(c, Gguf.architecture(f) == "demo", "architecture")
Check.check(c, Enum.at(Gguf.tensor(f, "w"), 2) == 2.0, "tensor")
Check.check(c, Gguf.info(f, "w")["dtype"] == "F32", "info")
File.rm!(path)

Check.check(c, hd(Rlx.devices()) == "cpu" and Rlx.is_available("cpu"), "cpu")
{g, _} = loss_graph.()
runner = Runner.new(g, policy: %{only: ["cpu"]})
Runner.set_param(runner, "w", [1, 0, 0, 1])
res = Runner.run(runner, feed)
Check.check(c, res.device == "cpu" and res.outputs == [[2.5]], "runner")
Check.check(c, Group.all_reduce(Group.connect(0, 1), [2, 3], op: "max") == [2.0, 3.0], "all_reduce")
IO.puts("test_rlx: #{:counters.get(c, 1)} checks passed")
