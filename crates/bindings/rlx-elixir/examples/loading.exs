# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
# into a compiled graph, and quantizing in memory.
#   mix run examples/loading.exs
alias Rlx.{Compiled, Gguf, GgufWriter, Graph, Session}

tmp = System.tmp_dir!()
w = for i <- 0..63, do: rem(i, 9) / 4 - 1

gguf = Path.join(tmp, "rlx-example-elixir.gguf")
writer = GgufWriter.new()
GgufWriter.add_f32(writer, "proj.weight", [8, 8], "F32", w)
GgufWriter.add_f32(writer, "proj.weight_q8", [8, 8], "Q8_0", w)
GgufWriter.write(writer, gguf, architecture: "demo", metadata: %{"demo.layers" => 1})
f = Gguf.open(gguf)
IO.puts("gguf architecture: #{Gguf.architecture(f)}")
IO.puts("gguf tensors: #{Enum.join(Gguf.tensor_names(f), " ")}")
exact = Gguf.tensor(f, "proj.weight")
q8 = Gguf.tensor(f, "proj.weight_q8")
maxd = Enum.zip_with(exact, q8, &abs(&1 - &2)) |> Enum.max()
IO.puts("gguf f32 exact: #{exact == w} q8 max|Δ|: #{:erlang.float_to_binary(maxd, decimals: 4)}")

g = Graph.new(name: "proj")
x = Graph.input(g, "x", [1, 8])
Graph.set_outputs(g, [Graph.matmul(g, x, Graph.param(g, "proj.weight", [8, 8]))])
c = Session.compile(Session.new(), g)
Compiled.set_param(c, "proj.weight", exact)
[y] = Compiled.run(c, %{"x" => List.duplicate(1, 8)})
IO.puts("gguf -> graph: #{y |> Enum.take(3) |> Enum.map_join(" ", &:erlang.float_to_binary(&1, decimals: 2))}")

st = Path.join(tmp, "rlx-example-elixir.safetensors")
header = ~s({"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}})
header = header <> String.duplicate(" ", rem(8 - rem(byte_size(header), 8), 8))
File.write!(st, [<<byte_size(header)::little-64>>, header, for(v <- w, do: <<v::float-little-32>>)])
loaded = Rlx.load_weights("safetensors", st)
IO.puts("safetensors: [proj.weight] exact: #{loaded["proj.weight"] == w}")

packed = Rlx.quantize(w, "Q8_0")
back = Rlx.dequantize(packed, "Q8_0", count: length(w))
IO.puts("quantize q8_0: #{length(w) * 4} -> #{byte_size(packed)} bytes, round trip #{length(back)} values")
File.rm!(gguf)
File.rm!(st)
IO.puts("argmax: #{Rlx.argmax([0.1, 2, 0.3])}")
