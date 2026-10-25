# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
# into a compiled graph, and quantizing in memory. RLX_TOKENIZER=<tokenizer.json>
# adds the tokenizer.
#
#     julia --project=crates/bindings/rlx-julia crates/bindings/rlx-julia/examples/loading.jl

using RLX
using Printf

w = Float32[(i % 9) / 4 - 1 for i in 0:63]
gguf = joinpath(tempdir(), "rlx-example-jl.gguf")
writer = RLX.GgufWriter()
RLX.add_f32(writer, "proj.weight", [8, 8], "F32", w)
RLX.add_f32(writer, "proj.weight_q8", [8, 8], "Q8_0", w)
RLX.write(writer, gguf; architecture = "demo", metadata = Dict("demo.layers" => 1))
f = RLX.open(RLX.Gguf, gguf)
println("gguf architecture: ", RLX.architecture(f))
println("gguf tensors: ", join(RLX.tensor_names(f), " "))
exact = RLX.tensor(f, "proj.weight"); q8 = RLX.tensor(f, "proj.weight_q8")
@printf("gguf f32 exact: %s q8 max|Δ|: %.4f\n", exact == w, maximum(abs.(exact .- q8)))

g = RLX.Graph(name = "proj")
x = RLX.input(g, "x", [1, 8])
RLX.set_outputs(g, [RLX.matmul(g, x, RLX.param(g, "proj.weight", [8, 8]))])
c = RLX.compile(RLX.Session(), g)
RLX.set_param(c, "proj.weight", exact)
println("gguf -> graph: ", join([@sprintf("%.2f", v) for v in RLX.run(c, Dict("x" => ones(Float32, 8)))[1][1:3]], " "))

st = joinpath(tempdir(), "rlx-example-jl.safetensors")
header = "{\"proj.weight\":{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,$(4 * length(w))]}}"
header *= " "^(mod(-length(header), 8))
Base.open(st, "w") do io
    Base.write(io, UInt64(length(header))); Base.write(io, header); Base.write(io, w)
end
loaded = RLX.load_weights("safetensors", st)
println("safetensors: ", collect(keys(loaded)), " exact: ", loaded["proj.weight"] == w)

packed = RLX.quantize(w, "Q8_0")
println("quantize q8_0: $(4 * length(w)) -> $(length(packed)) bytes, round trip $(length(RLX.dequantize(packed, "Q8_0"; count = length(w)))) values")
rm(gguf); rm(st)

tok_path = get(ENV, "RLX_TOKENIZER", nothing)
if tok_path === nothing
    println("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)")
else
    tok = RLX.load(RLX.Tokenizer, tok_path)
    ids = RLX.encode(tok, "Hello from Julia")
    println("tokenizer: $(length(ids)) ids -> ", repr(RLX.decode(tok, ids)))
end
println("argmax: ", RLX.argmax(Float32[0.1, 2, 0.3]))
