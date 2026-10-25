# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# frozen_string_literal: true

# Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
# into a compiled graph, and quantizing in memory. RLX_TOKENIZER=<tokenizer.json>
# adds the tokenizer.
#
#     ruby -Icrates/bindings/rlx-ruby/lib crates/bindings/rlx-ruby/examples/loading.rb

require "rlx"
require "json"
require "tmpdir"

w = Array.new(64) { |i| (i % 9) / 4.0 - 1 }
gguf = File.join(Dir.tmpdir, "rlx-example-rb.gguf")
writer = RLX::GgufWriter.new
writer.add_f32("proj.weight", [8, 8], "F32", w)
writer.add_f32("proj.weight_q8", [8, 8], "Q8_0", w)
writer.write(gguf, architecture: "demo", metadata: { "demo.layers" => 1 })
f = RLX::Gguf.open(gguf)
puts "gguf architecture: #{f.architecture}"
puts "gguf tensors: #{f.tensor_names.join(' ')}"
exact = f.tensor("proj.weight")
q8 = f.tensor("proj.weight_q8")
puts "gguf f32 exact: #{exact == w} q8 max|Δ|: #{format('%.4f', exact.zip(q8).map { |a, b| (a - b).abs }.max)}"

g = RLX::Graph.new(name: "proj")
x = g.input("x", [1, 8])
g.set_outputs([g.matmul(x, g.param("proj.weight", [8, 8]))])
c = RLX::Session.new.compile(g)
c.set_param("proj.weight", exact)
puts "gguf -> graph: #{c.run({ 'x' => [1.0] * 8 })[0].first(3).map { |v| format('%.2f', v) }.join(' ')}"

st = File.join(Dir.tmpdir, "rlx-example-rb.safetensors")
header = JSON.generate({ "proj.weight" => { dtype: "F32", shape: [8, 8], data_offsets: [0, w.size * 4] } })
header += " " * (-header.bytesize % 8)
File.binwrite(st, [header.bytesize].pack("Q<") + header + w.pack("e*"))
loaded = RLX.load_weights("safetensors", st)
puts "safetensors: #{loaded.keys} exact: #{loaded['proj.weight'] == w}"

packed = RLX.quantize(w, "Q8_0")
puts "quantize q8_0: #{w.size * 4} -> #{packed.bytesize} bytes, round trip #{RLX.dequantize(packed, 'Q8_0', count: w.size).size} values"
File.delete(gguf, st)

if (tok_path = ENV["RLX_TOKENIZER"])
  tok = RLX::Tokenizer.load(tok_path)
  ids = tok.encode("Hello from Ruby")
  puts "tokenizer: #{ids.size} ids -> #{tok.decode(ids).inspect}"
else
  puts "tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)"
end
puts "argmax: #{RLX.argmax([0.1, 2, 0.3])}"
