-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- Picking a backend: what this build has, what suits a graph, and the ways to
-- choose — by name, by cost model, by policy, by benchmark, per run.
--   lua examples/backends.lua

local rlx = require("rlx")

local function model()
  local g = rlx.Graph.new({ name = "mlp" })
  local x = g:input("x", { 4, 16 })
  g:set_outputs({ g:tanh(g:matmul(x, g:param("w", { 16, 16 }))) })
  return g
end
local xs, w = {}, {}
for i = 0, 63 do xs[i + 1] = (i % 7) / 7 end
for i = 0, 255 do w[i + 1] = (i % 5) / 10 - 0.2 end
local x = { x = xs }

local available = rlx.devices()
print("devices: " .. table.concat(available, " "))
print("metal available: " .. tostring(rlx.is_available("metal")))
print("fastest for graph: " .. rlx.fastest_device_for(model()))
local reference
for _, device in ipairs(available) do
  local c = rlx.Session.new({ device = device }):compile(model())
  c:set_param("w", w)
  local y = c:run(x)[1]
  reference = reference or y
  local d = 0
  for i = 1, #y do d = math.max(d, math.abs(y[i] - reference[i])) end
  print(string.format("session %s: max|Δ| vs first = %.2e", device, d))
end
local flex = rlx.FlexibleSession.new({ policy = { prefer = { "metal", "cpu" } } })
print("flexible session picked: " .. flex:compile(model()):device())
local runner = rlx.Runner.new(model())
runner:set_param("w", w)
print("runner warmed: " .. table.concat(runner:warm_all(), " "))
for _, row in ipairs(runner:benchmark(x, { runs = 3 })) do print("  bench " .. row.device .. ": exec_ms=" .. row.exec_ms) end
print("runner ran on: " .. runner:run(x).device)
print("run_try chain: " .. runner:run_try({ "metal", "cpu" }, x).device)
local router = rlx.Router.new(model())
router:set_param("w", w)
print("router ran on: " .. router:run_chain(x).device)
