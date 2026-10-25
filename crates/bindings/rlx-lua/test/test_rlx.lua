-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- The Lua binding against the real library; errors out on the first failure.

local rlx = require("rlx")
local checks = 0
local function check(ok, what) checks = checks + 1; assert(ok, "failed: " .. what) end

local function loss_graph()
  local g = rlx.Graph.new({ name = "mse" })
  local x, y = g:input("x", { 1, 2 }), g:input("y", { 1, 2 })
  local w = g:param("w", { 2, 2 })
  local d = g:sub(g:matmul(x, w), y)
  g:set_outputs({ g:mean(g:mul(d, d), { 0, 1 }) })
  return g, w
end

do -- inference
  local g = loss_graph()
  local m = rlx.Session.new():compile(g)
  check(g:is_consumed(), "compile consumes")
  m:set_param("w", { 1, 0, 0, 1 })
  check(m:run({ x = { 1, 2 }, y = { 0, 0 } })[1][1] == 2.5, "loss 2.5")
  check(m:device() == "cpu", "device")
end
do -- errors
  local ok, e = pcall(function() rlx.Graph.new():relu(42) end)
  check(not ok and e.kind == "range", "range error")
end
do -- training
  local g, w = loss_graph()
  local bwd = rlx.Session.new():compile(rlx.grad(g, { w }))
  bwd:set_param("w", { 1, 0, 0, 1 })
  check(bwd:run({ x = { 1, 2 }, y = { 0, 0 }, d_output = { 1 } })[2][4] == 4, "dw")
  local t = rlx.Trainer.new(g:clone(), { "w" }, { w = { 0, 0, 0, 0 } }, { options = { optimizer = { kind = "adamw", lr = 0.1 } } })
  local r = t:run(300, function() return { x = { 1, 1 }, y = { 1, -1 } } end, { on_step = function(_, l) return l > 1e-6 end })
  check(r.last_loss < 1e-4, "trainer converges")
  local opt = rlx.Optimizer.new({ kind = "sgd", options = { lr = 0.5 } })
  local p = { p = { 1, -1 } }
  opt:step(p, { p = { 2, -2 } })
  check(p.p[1] == 0 and p.p[2] == 0, "optimizer in place")
end
do -- loading
  local path = (os.getenv("TMPDIR") or ".") .. "/rlx-lua-test.gguf"
  local w = rlx.GgufWriter.new()
  w:add_f32("w", { 4 }, "F32", { 0.5, -1, 2, 0.25 })
  check(w:write(path, { architecture = "demo" }) == 1, "write")
  local f = rlx.Gguf.open(path)
  check(f:architecture() == "demo", "architecture")
  check(f:tensor("w")[3] == 2, "tensor")
  check(f:info("w").dtype == "F32", "info")
  os.remove(path)
end
do -- backends and a one-rank mesh
  check(rlx.devices()[1] == "cpu" and rlx.is_available("cpu"), "cpu")
  local g = loss_graph()
  local r = rlx.Runner.new(g, { policy = { only = { "cpu" } } })
  r:set_param("w", { 1, 0, 0, 1 })
  local res = r:run({ x = { 1, 2 }, y = { 0, 0 } })
  check(res.device == "cpu" and res.outputs[1][1] == 2.5, "runner")
  local mesh = rlx.Group.connect(0, 1)
  check(mesh:all_reduce({ 2, 3 }, { op = "max" })[2] == 3, "all_reduce")
end
print("test_rlx: " .. checks .. " checks passed")
