# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# frozen_string_literal: true

# The Ruby binding against the real library:
#     ruby -Icrates/bindings/rlx-ruby/lib crates/bindings/rlx-ruby/test/test_rlx.rb

require "minitest/autorun"
require "tmpdir"
require "rlx"

class TestRLX < Minitest::Test
  BATCH = { "x" => [1.0, 1.0], "y" => [1.0, -1.0] }.freeze

  # loss = mean((x·w − y)²), x/y [1, 2], w [2, 2].
  def loss_graph
    g = RLX::Graph.new(name: "mse")
    x = g.input("x", [1, 2])
    y = g.input("y", [1, 2])
    w = g.param("w", [2, 2])
    d = g.sub(g.matmul(x, w), y)
    g.set_outputs([g.mean(g.mul(d, d), [0, 1])])
    [g, w]
  end

  def test_inference
    g, = loss_graph
    m = RLX::Session.new(device: "cpu").compile(g)
    assert g.is_consumed
    m.set_param("w", [1, 0, 0, 1])
    assert_in_delta 2.5, m.run({ "x" => [1, 2], "y" => [0, 0] })[0][0], 1e-6
    assert_equal "cpu", m.device
    assert_equal ["f32"], m.output_dtypes
  end

  def test_errors_carry_kind_and_message
    e = assert_raises(RLX::Error) { RLX::Graph.new.relu(42) }
    assert_equal :range, e.kind
    assert_match(/does not exist/, e.message)
    assert_raises(RLX::Error) { RLX::Session.new(device: "no-such-device") }
  end

  def test_training
    g, w = loss_graph
    bwd = RLX::Session.new.compile(RLX.grad(g, [w]))
    bwd.set_param("w", [1, 0, 0, 1])
    dw = bwd.run({ "x" => [1, 2], "y" => [0, 0], "d_output" => [1] })[1]
    assert_equal [1.0, 2.0, 2.0, 4.0], dw

    t = RLX::Trainer.new(g.clone, ["w"], { "w" => [0, 0, 0, 0] },
                         options: { optimizer: { kind: "adamw", lr: 0.1 } })
    summary = t.run(300, on_step: ->(_s, l) { l > 1e-6 }) { BATCH }
    assert_operator summary["last_loss"], :<, 1e-4
    path = File.join(Dir.tmpdir, "rlx-rb-ck.gguf")
    assert_operator t.save(path), :>=, 1
    t2 = RLX::Trainer.new(g.clone, ["w"], { "w" => [0, 0, 0, 0] })
    assert_equal summary["steps"], t2.load(path)["step"]
    File.delete(path)
    assert_equal 1, t2.freeze_params
    assert_equal ["w"], t2.frozen

    opt = RLX::Optimizer.new(kind: "sgd", options: { lr: 0.5 })
    params = { "p" => [1.0, -1.0] }
    opt.step(params, { "p" => [2.0, -2.0] })
    assert_equal [0.0, 0.0], params["p"]
  end

  def test_loading
    path = File.join(Dir.tmpdir, "rlx-rb.gguf")
    w = [0.5, -1.0, 2.0, 0.25]
    writer = RLX::GgufWriter.new
    writer.add_f32("w", [4], "F32", w)
    assert_equal 1, writer.write(path, architecture: "demo", metadata: { "demo.layers" => 1 })
    f = RLX::Gguf.open(path)
    assert_equal "demo", f.architecture
    assert_equal w, f.tensor("w")
    assert_equal "F32", f.info("w")["dtype"]
    assert_equal 1, f.metadata_value("demo.layers")
    File.delete(path)
    packed = RLX.quantize(w * 8, "Q8_0")
    assert_equal 32, RLX.dequantize(packed, "Q8_0").size
  end

  def test_backends
    assert_includes RLX.devices, "cpu"
    assert RLX.is_available("cpu")
    g, = loss_graph
    assert_equal "cpu", RLX.fastest_device_for(g)
    r = RLX::Runner.new(g, policy: { only: ["cpu"] })
    r.set_param("w", [1, 0, 0, 1])
    res = r.run({ "x" => [1, 2], "y" => [0, 0] })
    assert_equal "cpu", res[:device]
    assert_in_delta 2.5, res[:outputs][0][0], 1e-6
    assert_equal ["cpu"], r.warm_all
  end

  def test_a_two_rank_mesh
    peers = RLX.loopback_peers(2)
    worker = Thread.new do
      g = RLX::Group.connect(1, 2, options: { peers: peers })
      g.all_reduce([3.0], op: "max")
      g.serve_worker(options: { steps: 1 })
    end
    g = RLX::Group.connect(0, 2, options: { peers: peers })
    assert_equal [5.0], g.all_reduce([5.0], op: "max")
    assert_equal 1, g.collect_caps.size
    s = RLX::Graph.new(name: "double")
    x = s.input("x", [2])
    s.set_outputs([s.add(x, x)])
    g.ship_stage(1, s, { device: "cpu" })
    g.send_activation(1, [1.0, 2.0])
    assert_equal [2.0, 4.0], g.recv_activation(1)
    assert_equal 1, worker.value["activations"]
  end
end
