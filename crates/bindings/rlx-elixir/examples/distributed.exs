# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Distributed: a three-rank mesh on loopback, one BEAM process per rank (the
# calls run on dirty schedulers, so ranks block independently); across
# machines, one node per host with the same peers. Rank 0 coordinates:
# collectives, pipelined inference (a stage shipped per worker), data-parallel
# training where every replica ends identical.
#   mix run examples/distributed.exs
alias Rlx.{Graph, Group}

world = 3
per_rank = 16
peers = Rlx.loopback_peers(world)
tmp = Path.join(System.tmp_dir!(), "rlx-dist-elixir-#{System.unique_integer([:positive])}")
File.mkdir_p!(tmp)

blob = fn name, values ->
  path = Path.join(tmp, name)
  File.write!(path, for(v <- values, do: <<v::float-little-32>>))
  "file://" <> path
end

collectives = fn g, rank ->
  total = Group.all_reduce(g, [rank + 1], op: "sum")
  gathered = Group.all_gather(g, [rank + 1])
  shared = Group.broadcast(g, 0, [if(rank == 0, do: 42, else: 0)])

  if rank == 0 do
    IO.puts("mesh: rank #{Group.rank(g)} of #{Group.world(g)}, leader: #{Group.is_leader(g)}")
    IO.puts("all_reduce sum: #{Enum.join(total, ",")}")
    IO.puts("all_gather: #{Enum.join(gathered, ",")}")
    IO.puts("broadcast: #{Enum.join(shared, ",")}")
  end
end

workers =
  for rank <- 1..(world - 1) do
    Task.async(fn ->
      g = Group.connect(rank, world, options: %{peers: peers, device: "cpu"})
      collectives.(g, rank)
      Group.serve_worker(g, options: %{steps: 2})
      Group.serve_trainer(g)["params"]["w"]
    end)
  end

g = Group.connect(0, world, options: %{peers: peers, device: "cpu"})
collectives.(g, 0)

stage_w = [for(i <- 0..15, do: rem(i * 5, 7) / 7 - 0.3), for(i <- 0..15, do: rem(i * 3, 5) / 5 - 0.2)]
IO.puts("workers: #{Enum.map_join(Group.collect_caps(g), ", ", &"rank #{&1["rank"]} on #{&1["platform"]}")}")
stage = Graph.new(name: "stage")
Graph.set_outputs(stage, [Graph.relu(stage, Graph.matmul(stage, Graph.input(stage, "x", [1, 4]), Graph.param(stage, "w", [4, 4])))])

for r <- 1..(world - 1),
    do: Group.ship_stage(g, r, stage, %{device: "cpu", weights: %{w: blob.("w#{r}.bin", Enum.at(stage_w, rem(r - 1, 2)))}})

for xv <- [[1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]] do
  Group.send_activation(g, 1, xv)
  IO.puts("pipeline: #{Enum.map_join(Group.recv_activation(g, world - 1), " ", &:erlang.float_to_binary(&1, decimals: 3))}")
end

Group.collect_caps(g)
lr = Graph.new(name: "lr")
bx = Graph.input(lr, "X", [4, 3])
by = Graph.input(lr, "y", [4, 1])
wv = Graph.param(lr, "w", [3, 1])
d = Graph.sub(lr, Graph.matmul(lr, bx, wv), by)
Graph.set_outputs(lr, [Graph.mean(lr, Graph.mul(lr, d, d), [0, 1])])
bwd = Rlx.grad(lr, [wv])
w_true = [1.5, -2.0, 0.5]
n = per_rank * world
x_all = for i <- 0..(n * 3 - 1), do: rem(i * 37, 23) / 11.5 - 1
y_all = x_all |> Enum.chunk_every(3) |> Enum.map(fn r -> Enum.zip_with(r, w_true, &(&1 * &2)) |> Enum.sum() end)
uris = %{x: blob.("X.bin", x_all), y: blob.("y.bin", y_all), w: blob.("w0.bin", [0, 0, 0])}

spec = fn r ->
  %{
    params: [%{name: "w", uri: uris.w}],
    data: [
      %{input: "X", uri: uris.x, elem: 3, shard_start: r * per_rank, shard_len: per_rank},
      %{input: "y", uri: uris.y, elem: 1, shard_start: r * per_rank, shard_len: per_rank}
    ],
    batch: 4,
    lr_per_epoch: List.duplicate(0.2, 40),
    device: "cpu"
  }
end

for r <- 1..(world - 1), do: Group.ship_train(g, r, bwd, spec.(r))
mine = Group.run_train(g, bwd, spec.(0))

IO.puts(
  "training loss: #{:erlang.float_to_binary(mine["metrics"]["first_loss"] * 1.0, decimals: 4)} -> " <>
    "#{:io_lib.format("~.2e", [mine["metrics"]["last_loss"] * 1.0])}"
)

w0 = mine["params"]["w"]
IO.puts("rank 0 w: #{Enum.map_join(w0, " ", &:erlang.float_to_binary(&1 * 1.0, decimals: 3))} (true: 1.5 -2 0.5)")

for {t, r} <- Enum.with_index(workers, 1) do
  theirs = Task.await(t, :infinity)
  IO.puts("rank #{r} w agrees: #{Enum.zip_with(theirs, w0, &(abs(&1 - &2) < 1.0e-5)) |> Enum.all?()}")
end

File.rm_rf!(tmp)
