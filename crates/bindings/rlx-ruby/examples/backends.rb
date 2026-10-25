# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# frozen_string_literal: true

# Picking a backend: what this build has, what suits a graph, and the ways to
# choose — by name, by cost model, by policy, by benchmark, per run.
#
#     ruby -Icrates/bindings/rlx-ruby/lib crates/bindings/rlx-ruby/examples/backends.rb

require "rlx"

def model
  g = RLX::Graph.new(name: "mlp")
  x = g.input("x", [4, 16])
  g.set_outputs([g.tanh(g.matmul(x, g.param("w", [16, 16])))])
  g
end

x = { "x" => Array.new(64) { |i| (i % 7) / 7.0 } }
w = Array.new(256) { |i| (i % 5) / 10.0 - 0.2 }

available = RLX.devices
puts "devices: #{available.join(' ')}"
puts "metal available: #{RLX.is_available('metal')}"
puts "fastest for graph: #{RLX.fastest_device_for(model)}"
reference = nil
available.each do |device|
  c = RLX::Session.new(device: device).compile(model)
  c.set_param("w", w)
  y = c.run(x)[0]
  reference ||= y
  puts "session #{device}: max|Δ| vs first = #{format('%.2e', y.zip(reference).map { |a, b| (a - b).abs }.max)}"
end
puts "flexible session picked: #{RLX::FlexibleSession.new(policy: { prefer: %w[metal cpu] }).compile(model).device}"
runner = RLX::Runner.new(model)
runner.set_param("w", w)
puts "runner warmed: #{runner.warm_all.join(' ')}"
runner.benchmark(x, runs: 3).each { |row| puts "  bench #{row['device']}: exec_ms=#{row['exec_ms']}" }
puts "runner ran on: #{runner.run(x)[:device]}"
puts "run_try chain: #{runner.run_try(%w[metal cpu], x)[:device]}"
router = RLX::Router.new(model)
router.set_param("w", w)
puts "router ran on: #{router.run_chain(x)[:device]}"
