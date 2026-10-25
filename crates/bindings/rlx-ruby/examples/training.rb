# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# frozen_string_literal: true

# Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
# the same fit by hand with `grad` and an Optimizer.
#
#     ruby -Icrates/bindings/rlx-ruby/lib crates/bindings/rlx-ruby/examples/training.rb [device]

require "rlx"
require "tmpdir"

W_TRUE = [2.0, -1.0, 0.5].freeze
BATCH = 8

def data(step)
  x = []
  y = []
  BATCH.times do |i|
    row = Array.new(3) { |k| ((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1 }
    x.concat(row)
    y << row.zip(W_TRUE).sum { |a, b| a * b }
  end
  { "x" => x, "y" => y }
end

fmt = ->(v) { v.map { |a| format("%.3f", a) }.join(" ") }
device = ARGV[0] || "cpu"
g = RLX::Graph.new(name: "regression")
x = g.input("x", [BATCH, 3])
y = g.input("y", [BATCH, 1])
w = g.param("w", [3, 1])
d = g.sub(g.matmul(x, w), y)
g.set_outputs([g.mean(g.mul(d, d), [0, 1])])

trainer = RLX::Trainer.new(g.clone, ["w"], { "w" => [0, 0, 0] },
                           options: { device: device, optimizer: { kind: "adamw", lr: 0.1 } })
first = trainer.evaluate(data(0))
trainer.run(400, lr: ->(s) { 0.1 * 0.5 * (1 + Math.cos(Math::PI * s / 400)) },
                 on_step: ->(_s, loss) { loss > 1e-6 }) { |s| data(s) }
puts "trainer loss: #{format('%.4f', first)} -> #{format('%.6f', trainer.evaluate(data(0)))} in #{trainer.steps} steps"
puts "trainer w: #{fmt.call(trainer.params['w'])}"

path = File.join(Dir.tmpdir, "rlx-example-train-rb.gguf")
written = trainer.save(path)
restored = RLX::Trainer.new(g.clone, ["w"], { "w" => [0, 0, 0] }, options: { device: device })
puts "checkpoint: #{written} tensors, restored step #{restored.load(path)['step']}"
File.delete(path)
restored.freeze_params(names: ["w"])
before = restored.params["w"]
restored.step(data(1))
puts "frozen: #{restored.frozen} unchanged: #{restored.params['w'] == before}"
restored.unfreeze_params

backward = RLX::Session.new(device: device).compile(RLX.grad(g, [w])) # outputs [loss, dw]
opt = RLX::Optimizer.new(kind: "sgd", options: { lr: 0.2 })
params = { "w" => [0.0, 0.0, 0.0] }
loss = 0
300.times do |step|
  backward.set_param("w", params["w"])
  loss_v, dw = backward.run(data(step).merge("d_output" => [1]))
  loss = loss_v[0]
  opt.step(params, { "w" => dw })
end
puts "manual loss: #{format('%.6f', loss)} w: #{fmt.call(params['w'])}"
