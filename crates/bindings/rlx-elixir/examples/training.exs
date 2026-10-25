# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
# the same fit by hand with grad and an Optimizer.
#   mix run examples/training.exs [device]
alias Rlx.{Compiled, Graph, Optimizer, Session, Trainer}

device = List.first(System.argv()) || "cpu"
w_true = [2.0, -1.0, 0.5]
batch = 8

data = fn step ->
  rows =
    for i <- 0..(batch - 1) do
      for k <- 0..2, do: rem(step * 31 + i * 7 + k * 3, 17) / 8.5 - 1
    end

  %{
    "x" => List.flatten(rows),
    "y" => Enum.map(rows, fn r -> Enum.zip_with(r, w_true, &(&1 * &2)) |> Enum.sum() end)
  }
end

fmt = fn v -> Enum.map_join(v, " ", &:erlang.float_to_binary(&1 * 1.0, decimals: 3)) end

g = Graph.new(name: "regression")
x = Graph.input(g, "x", [batch, 3])
y = Graph.input(g, "y", [batch, 1])
w = Graph.param(g, "w", [3, 1])
d = Graph.sub(g, Graph.matmul(g, x, w), y)
Graph.set_outputs(g, [Graph.mean(g, Graph.mul(g, d, d), [0, 1])])

trainer =
  Trainer.new(Graph.clone(g), ["w"], %{"w" => [0, 0, 0]},
    options: %{device: device, optimizer: %{kind: "adamw", lr: 0.1}}
  )

first = Trainer.evaluate(trainer, data.(0))

Trainer.run(trainer, 400, data,
  lr: fn s -> 0.1 * 0.5 * (1 + :math.cos(:math.pi() * s / 400)) end,
  on_step: fn _s, loss -> loss > 1.0e-6 end
)

IO.puts(
  "trainer loss: #{:erlang.float_to_binary(first, decimals: 4)} -> " <>
    "#{:erlang.float_to_binary(Trainer.evaluate(trainer, data.(0)), decimals: 6)} in #{Trainer.steps(trainer)} steps"
)

IO.puts("trainer w: #{fmt.(Trainer.params(trainer)["w"])}")

path = Path.join(System.tmp_dir!(), "rlx-example-train-elixir.gguf")
written = Trainer.save(trainer, path)
restored = Trainer.new(Graph.clone(g), ["w"], %{"w" => [0, 0, 0]}, options: %{device: device})
IO.puts("checkpoint: #{written} tensors, restored step #{Trainer.load(restored, path)["step"]}")
File.rm!(path)
Trainer.freeze(restored, names: ["w"])
before = fmt.(Trainer.params(restored)["w"])
Trainer.step(restored, data.(1))
IO.puts("frozen: #{inspect(Trainer.frozen(restored))} unchanged: #{fmt.(Trainer.params(restored)["w"]) == before}")
Trainer.unfreeze(restored)

backward = Session.compile(Session.new(device: device), Rlx.grad(g, [w]))
opt = Optimizer.new(kind: "sgd", options: %{lr: 0.2})

{params, loss} =
  Enum.reduce(0..299, {%{"w" => [0, 0, 0]}, 0}, fn step, {p, _} ->
    Compiled.set_param(backward, "w", p["w"])
    [[l], dw] = Compiled.run(backward, Map.put(data.(step), "d_output", [1]))
    {Optimizer.step(opt, p, %{"w" => dw}), l}
  end)

IO.puts("manual loss: #{:erlang.float_to_binary(loss, decimals: 6)} w: #{fmt.(params["w"])}")
