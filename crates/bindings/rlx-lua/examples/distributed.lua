-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- Distributed: Lua is single-threaded, so each rank is a process; start one
-- per rank with the same peers (here three on loopback; across machines, one
-- per host):
--   lua examples/distributed.lua 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
--   lua examples/distributed.lua 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
--   lua examples/distributed.lua 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
-- Rank 0 coordinates: collectives, pipelined inference (a stage shipped per
-- worker), data-parallel training. With no arguments it is a mesh of one.

local rlx = require("rlx")
local rank, world = tonumber(arg[1] or 0), tonumber(arg[2] or 1)
local peers
if arg[3] then peers = {}; for p in arg[3]:gmatch("[^,]+") do peers[#peers + 1] = p end end
local tmp = os.getenv("TMPDIR") or "."
local PER_RANK = 16
local blobs = {}

local function blob(name, values)
  local path = tmp .. "/rlx-dist-lua-" .. name
  local f = assert(io.open(path, "wb"))
  for _, v in ipairs(values) do f:write(string.pack("<f", v)) end
  f:close()
  blobs[#blobs + 1] = path
  return "file://" .. path
end

local g = rlx.Group.connect(rank, world, { options = { peers = peers, device = "cpu" } })
local sum = g:all_reduce({ rank + 1 }, { op = "sum" })
local gathered = g:all_gather({ rank + 1 })
local shared = g:broadcast(0, { rank == 0 and 42 or 0 }) -- the root's length
if rank == 0 then
  print(string.format("mesh: rank %d of %d, leader: %s", g:rank(), g:world(), tostring(g:is_leader())))
  print("all_reduce sum: " .. table.concat(sum, ","))
  print("all_gather: " .. table.concat(gathered, ","))
  print("broadcast: " .. table.concat(shared, ","))
end
if world == 1 then
  print("a mesh of one: pass `rank world peers` to start more ranks")
  return
end
if rank ~= 0 then
  g:serve_worker({ options = { steps = 2 } })
  g:send(0, 1, g:serve_trainer().params.w) -- rank 0 compares
  return
end

local stage_w = { {}, {} }
for i = 0, 15 do stage_w[1][i + 1] = (i * 5 % 7) / 7 - 0.3; stage_w[2][i + 1] = (i * 3 % 5) / 5 - 0.2 end
local caps = {}
for _, c in ipairs(g:collect_caps()) do caps[#caps + 1] = "rank " .. c.rank .. " on " .. c.platform end
print("workers: " .. table.concat(caps, ", "))
local stage = rlx.Graph.new({ name = "stage" })
stage:set_outputs({ stage:relu(stage:matmul(stage:input("x", { 1, 4 }), stage:param("w", { 4, 4 }))) })
for r = 1, world - 1 do g:ship_stage(r, stage, { device = "cpu", weights = { w = blob("w" .. r .. ".bin", stage_w[(r - 1) % 2 + 1]) } }) end
for _, xv in ipairs({ { 1, -1, 0.5, 2 }, { 0.25, 0.5, -0.75, 1 } }) do
  g:send_activation(1, xv)
  local yv = g:recv_activation(world - 1)
  print(string.format("pipeline: %.3f %.3f %.3f %.3f", yv[1], yv[2], yv[3], yv[4]))
end

g:collect_caps()
local lr = rlx.Graph.new({ name = "lr" })
local X, Y, wv = lr:input("X", { 4, 3 }), lr:input("y", { 4, 1 }), lr:param("w", { 3, 1 })
local d = lr:sub(lr:matmul(X, wv), Y)
lr:set_outputs({ lr:mean(lr:mul(d, d), { 0, 1 }) })
local bwd = rlx.grad(lr, { wv })
local w_true, n = { 1.5, -2, 0.5 }, PER_RANK * world
local x_all, y_all = {}, {}
for i = 0, n * 3 - 1 do x_all[i + 1] = (i * 37 % 23) / 11.5 - 1 end
for i = 0, n - 1 do
  local s = 0
  for k = 0, 2 do s = s + x_all[i * 3 + k + 1] * w_true[k + 1] end
  y_all[i + 1] = s
end
local uris = { X = blob("X.bin", x_all), y = blob("y.bin", y_all), w = blob("w0.bin", { 0, 0, 0 }) }
local lrs = {}
for i = 1, 40 do lrs[i] = 0.2 end
local function spec(r)
  return { params = { { name = "w", uri = uris.w } }, data = {
    { input = "X", uri = uris.X, elem = 3, shard_start = r * PER_RANK, shard_len = PER_RANK },
    { input = "y", uri = uris.y, elem = 1, shard_start = r * PER_RANK, shard_len = PER_RANK } },
    batch = 4, lr_per_epoch = lrs, device = "cpu" }
end
for r = 1, world - 1 do g:ship_train(r, bwd, spec(r)) end
local mine = g:run_train(bwd, spec(0))
print(string.format("training loss: %.4f -> %.2e", mine.metrics.first_loss, mine.metrics.last_loss))
local w0 = mine.params.w
print(string.format("rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)", w0[1], w0[2], w0[3]))
for r = 1, world - 1 do
  local theirs, agree = g:recv(r, 1), true
  for k = 1, #w0 do agree = agree and math.abs(theirs[k] - w0[k]) < 1e-5 end
  print(string.format("rank %d w agrees: %s", r, tostring(agree)))
end
for _, path in ipairs(blobs) do os.remove(path) end
