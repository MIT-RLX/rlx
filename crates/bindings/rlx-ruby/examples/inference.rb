# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# frozen_string_literal: true

# Inference: build a graph, compile it for a device, set its weights, run it.
#
#     ruby -Icrates/bindings/rlx-ruby/lib crates/bindings/rlx-ruby/examples/inference.rb [device]
#
# A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

require "rlx"

def weights(n, seed) = Array.new(n) { |i| ((i * 7919 + seed * 104_729) % 1000) / 1000.0 - 0.5 }

device = ARGV[0] || "cpu"
g = RLX::Graph.new(name: "classifier")
x = g.input("x", [1, 4])
hidden = g.relu(g.matmul(x, g.param("w1", [4, 8])))
g.set_outputs([g.softmax(g.matmul(hidden, g.param("w2", [8, 3])))])

model = RLX::Session.new(device: device).compile(g) # compile consumes the graph
puts "compiled on: #{model.device}"
model.set_params({ "w1" => weights(32, 1), "w2" => weights(24, 2) })
probs = model.run({ "x" => [0.5, -1, 2, 0.25] })[0]
puts "probs: #{probs.map { |v| format('%.4f', v) }.join(' ')}"
puts "probs sum: #{format('%.4f', probs.sum)}"
puts "output shapes: #{model.output_shapes} dtypes: #{model.output_dtypes}"
