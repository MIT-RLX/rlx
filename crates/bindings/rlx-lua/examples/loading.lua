-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
-- into a compiled graph, and quantizing in memory.
--   lua examples/loading.lua

local rlx = require("rlx")
local tmp = os.getenv("TMPDIR") or "."
local w = {}
for i = 0, 63 do w[i + 1] = (i % 9) / 4 - 1 end

local gguf = tmp .. "/rlx-example-lua.gguf"
local writer = rlx.GgufWriter.new()
writer:add_f32("proj.weight", { 8, 8 }, "F32", w)
writer:add_f32("proj.weight_q8", { 8, 8 }, "Q8_0", w)
writer:write(gguf, { architecture = "demo", metadata = { ["demo.layers"] = 1 } })
local f = rlx.Gguf.open(gguf)
print("gguf architecture: " .. f:architecture())
print("gguf tensors: " .. table.concat(f:tensor_names(), " "))
local exact, q8 = f:tensor("proj.weight"), f:tensor("proj.weight_q8")
local same, maxd = true, 0
for i = 1, 64 do same = same and exact[i] == w[i]; maxd = math.max(maxd, math.abs(exact[i] - q8[i])) end
print(string.format("gguf f32 exact: %s q8 max|Δ|: %.4f", tostring(same), maxd))

local g = rlx.Graph.new({ name = "proj" })
local x = g:input("x", { 1, 8 })
g:set_outputs({ g:matmul(x, g:param("proj.weight", { 8, 8 })) })
local c = rlx.Session.new():compile(g)
c:set_param("proj.weight", exact)
local y = c:run({ x = { 1, 1, 1, 1, 1, 1, 1, 1 } })[1]
print(string.format("gguf -> graph: %.2f %.2f %.2f", y[1], y[2], y[3]))

local st = tmp .. "/rlx-example-lua.safetensors"
local header = '{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}'
header = header .. string.rep(" ", (8 - #header % 8) % 8)
local file = assert(io.open(st, "wb"))
file:write(string.pack("<I8", #header), header)
for _, v in ipairs(w) do file:write(string.pack("<f", v)) end
file:close()
local loaded = rlx.load_weights("safetensors", st)
local exact2 = true
for i = 1, 64 do exact2 = exact2 and loaded["proj.weight"][i] == w[i] end
print("safetensors: [proj.weight] exact: " .. tostring(exact2))

local packed = rlx.quantize(w, "Q8_0")
print(string.format("quantize q8_0: %d -> %d bytes, round trip %d values", #w * 4, #packed, #rlx.dequantize(packed, "Q8_0", { count = #w })))
os.remove(gguf); os.remove(st)
print("argmax: " .. rlx.argmax({ 0.1, 2, 0.3 }))
