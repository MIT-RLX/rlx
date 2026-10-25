-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- Inference: build a graph, compile it for a device, set its weights, run it.
--   lua examples/inference.lua [device]
-- A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

local rlx = require("rlx")
local device = arg[1] or "cpu"

local function weights(n, seed)
  local w = {}
  for i = 0, n - 1 do w[i + 1] = ((i * 7919 + seed * 104729) % 1000) / 1000 - 0.5 end
  return w
end

local g = rlx.Graph.new({ name = "classifier" })
local x = g:input("x", { 1, 4 })
local hidden = g:relu(g:matmul(x, g:param("w1", { 4, 8 })))
g:set_outputs({ g:softmax(g:matmul(hidden, g:param("w2", { 8, 3 }))) })

local model <close> = rlx.Session.new({ device = device }):compile(g) -- consumes g
print("compiled on: " .. model:device())
model:set_params({ w1 = weights(32, 1), w2 = weights(24, 2) })
local probs = model:run({ x = { 0.5, -1, 2, 0.25 } })[1]
local parts, sum = {}, 0
for _, p in ipairs(probs) do parts[#parts + 1] = string.format("%.4f", p); sum = sum + p end
print("probs: " .. table.concat(parts, " "))
print(string.format("probs sum: %.4f", sum))
print("output shapes: " .. rlx.json.encode(model:output_shapes()) .. " dtypes: " .. rlx.json.encode(model:output_dtypes()))
