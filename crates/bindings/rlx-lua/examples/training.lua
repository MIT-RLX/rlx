-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
-- the same fit by hand with grad and an Optimizer.
--   lua examples/training.lua [device]

local rlx = require("rlx")
local device = arg[1] or "cpu"
local W_TRUE, BATCH = { 2, -1, 0.5 }, 8

local function data(step)
  local x, y = {}, {}
  for i = 0, BATCH - 1 do
    local s = 0
    for k = 0, 2 do
      local v = ((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1
      x[#x + 1] = v
      s = s + v * W_TRUE[k + 1]
    end
    y[#y + 1] = s
  end
  return { x = x, y = y }
end
local function fmt(v) return string.format("%.3f %.3f %.3f", v[1], v[2], v[3]) end

local g = rlx.Graph.new({ name = "regression" })
local x, y = g:input("x", { BATCH, 3 }), g:input("y", { BATCH, 1 })
local w = g:param("w", { 3, 1 })
local d = g:sub(g:matmul(x, w), y)
g:set_outputs({ g:mean(g:mul(d, d), { 0, 1 }) })

local trainer = rlx.Trainer.new(g:clone(), { "w" }, { w = { 0, 0, 0 } },
  { options = { device = device, optimizer = { kind = "adamw", lr = 0.1 } } })
local first = trainer:evaluate(data(0))
trainer:run(400, data, {
  lr = function(s) return 0.1 * 0.5 * (1 + math.cos(math.pi * s / 400)) end,
  on_step = function(_, loss) return loss > 1e-6 end,
})
print(string.format("trainer loss: %.4f -> %.6f in %d steps", first, trainer:evaluate(data(0)), trainer:steps()))
print("trainer w: " .. fmt(trainer:params().w))

local path = (os.getenv("TMPDIR") or ".") .. "/rlx-example-train-lua.gguf"
local written = trainer:save(path)
local restored = rlx.Trainer.new(g:clone(), { "w" }, { w = { 0, 0, 0 } }, { options = { device = device } })
print(string.format("checkpoint: %d tensors, restored step %d", written, restored:load(path).step))
os.remove(path)
restored:freeze({ names = { "w" } })
local before = fmt(restored:params().w)
restored:step(data(1))
print("frozen: " .. rlx.json.encode(restored:frozen()) .. " unchanged: " .. tostring(fmt(restored:params().w) == before))
restored:unfreeze()

local backward = rlx.Session.new({ device = device }):compile(rlx.grad(g, { w })) -- outputs [loss, dw]
local opt = rlx.Optimizer.new({ kind = "sgd", options = { lr = 0.2 } })
local params, loss = { w = { 0, 0, 0 } }, 0
for step = 0, 299 do
  backward:set_param("w", params.w)
  local feed = data(step)
  feed.d_output = { 1 }
  local outs = backward:run(feed)
  loss = outs[1][1]
  opt:step(params, { w = outs[2] })
end
print(string.format("manual loss: %.6f w: %s", loss, fmt(params.w)))
