# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# frozen_string_literal: true

# Distributed: three ranks as threads on loopback (calls release the GVL);
# across machines, one rank per host with the same `peers`. Collectives,
# pipelined inference (a stage shipped per worker), data-parallel training.
#
#     ruby -Icrates/bindings/rlx-ruby/lib crates/bindings/rlx-ruby/examples/distributed.rb

require "rlx"
require "tmpdir"
require "fileutils"

WORLD = 3
PER_RANK = 16
TMP = Dir.mktmpdir("rlx-dist-rb")

def blob(name, values)
  path = File.join(TMP, name)
  File.binwrite(path, values.pack("e*"))
  "file://#{path}"
end

def matvec(x, w) = (0...4).map { |j| [(0...4).sum { |i| x[i] * w[i * 4 + j] }, 0].max }

stage_w = { 1 => Array.new(16) { |i| (i * 5 % 7) / 7.0 - 0.3 }, 2 => Array.new(16) { |i| (i * 3 % 5) / 5.0 - 0.2 } }
w_true = [1.5, -2, 0.5]
x_all = Array.new(PER_RANK * WORLD * 3) { |i| (i * 37 % 23) / 11.5 - 1 }
y_all = Array.new(PER_RANK * WORLD) { |i| (0...3).sum { |k| x_all[i * 3 + k] * w_true[k] } }

def spec(rank, uris)
  start = rank * PER_RANK
  { params: [{ name: "w", uri: uris[:w] }],
    data: [{ input: "X", uri: uris[:X], elem: 3, shard_start: start, shard_len: PER_RANK },
           { input: "y", uri: uris[:y], elem: 1, shard_start: start, shard_len: PER_RANK }],
    batch: 4, lr_per_epoch: [0.2] * 40, device: "cpu" }
end

peers = RLX.loopback_peers(WORLD)
workers = (1...WORLD).map do |rank|
  Thread.new do
    g = RLX::Group.connect(rank, WORLD, options: { peers: peers, device: "cpu" })
    g.all_reduce([rank + 1.0], op: "sum")
    g.all_gather([rank + 1.0])
    g.broadcast(0, [0.0]) # the root's length; it overwrites
    g.serve_worker(options: { steps: 2 })
    g.serve_trainer["params"]["w"]
  end
end

g = RLX::Group.connect(0, WORLD, options: { peers: peers, device: "cpu" })
puts "mesh: rank #{g.rank} of #{g.world}, leader: #{g.is_leader}"
puts "all_reduce sum: #{g.all_reduce([1.0], op: 'sum')}"
puts "all_gather: #{g.all_gather([1.0])}"
puts "broadcast: #{g.broadcast(0, [42.0])}"

caps = g.collect_caps
puts "workers: #{caps.map { |c| "rank #{c['rank']} on #{c['platform']}" }.join(', ')}"
stage = RLX::Graph.new(name: "stage")
sx = stage.input("x", [1, 4])
stage.set_outputs([stage.relu(stage.matmul(sx, stage.param("w", [4, 4])))])
(1...WORLD).each { |r| g.ship_stage(r, stage, { device: "cpu", weights: { w: blob("w#{r}.bin", stage_w[r]) } }) }
[[1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]].each do |xv|
  g.send_activation(1, xv)
  yv = g.recv_activation(WORLD - 1)
  want = matvec(matvec(xv, stage_w[1]), stage_w[2])
  puts "pipeline max|Δ| vs local = #{format('%.1e', yv.zip(want).map { |a, b| (a - b).abs }.max)}"
end

g.collect_caps
lr = RLX::Graph.new(name: "lr")
xs = lr.input("X", [4, 3])
ys = lr.input("y", [4, 1])
wv = lr.param("w", [3, 1])
d = lr.sub(lr.matmul(xs, wv), ys)
lr.set_outputs([lr.mean(lr.mul(d, d), [0, 1])])
bwd = RLX.grad(lr, [wv])
uris = { X: blob("X.bin", x_all), y: blob("y.bin", y_all), w: blob("w0.bin", [0, 0, 0]) }
(1...WORLD).each { |r| g.ship_train(r, bwd, spec(r, uris)) }
mine = g.run_train(bwd, spec(0, uris))
w0 = mine["params"]["w"]
puts "training loss: #{format('%.4f', mine['metrics']['first_loss'])} -> #{format('%.2e', mine['metrics']['last_loss'])}"
puts "rank 0 w: #{w0.map { |v| format('%.3f', v) }.join(' ')} (true: 1.5 -2 0.5)"
workers.each_with_index do |t, i|
  puts "rank #{i + 1} w agrees: #{t.value.zip(w0).all? { |a, b| (a - b).abs < 1e-5 }}"
end
FileUtils.rm_rf(TMP)
